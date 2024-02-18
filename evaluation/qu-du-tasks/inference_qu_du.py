import argparse
import os
import json
import numpy as np
import torch
import random
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_sampling import DistributedEvalSampler
from inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from inference_tasks.query_description import QueryDescription
from inference_tasks.query_expansion import QueryExpansion
from inference_tasks.query_suggestion import QuerySuggestion
from inference_tasks.query_reformulation import QueryReformulation
from inference_tasks.query_clarification import QueryClarification
from inference_tasks.query_matching import QueryMatching
from inference_tasks.summarization import Summarization
from inference_tasks.fact_verification import FactVerification
from inference_tasks.query_intent_classification import QueryIntentClassification
from inference_tasks.reading_comprehension import ReadingComprehension
from inference_tasks.query_subtopic_generation import QuerySubtopicGeneration
from inference_tasks.conversational_qa import ConversationalQA
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib import tzip

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="model/", type=str)
parser.add_argument("--tokenizer_name", default="model/llama-2-7b-hf", type=str)
parser.add_argument("--setting", default="in-domain", type=str)
parser.add_argument("--n_shots", default="zero_shot", type=str)
parser.add_argument("--max_input_len", default=1792, type=int)
parser.add_argument("--max_output_len", default=256, type=int)
parser.add_argument("--seed", default=0, type=int)  
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--gpus', type=int,default=-1, help='num gpus per node')
parser.add_argument('--nr', type=int,default=0, help='ranking within the nodes') 
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, legacy=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def set_seed(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test_model(local_gpu_rank, args, tasks, label_lists, save_result=False):
    set_seed(args.seed)
    args.rank = args.nr * args.gpus + local_gpu_rank
    torch.cuda.set_device(local_gpu_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_cache=True, torch_dtype=torch.bfloat16)
    args.device = torch.device("cuda", local_gpu_rank)
    model.to(args.device)
    model.eval()

    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")

    if args.rank == 0:
        task_result = []
        fw = open(f"result/{time_string}.csv", "w", encoding="utf-8") 
        fw.write(f"Model,{args.model_name_or_path}\n")
    for task, label_list in tzip(tasks, label_lists, ncols=120, desc="# Tasks"):
        test_data = task.get_path()

        test_dataset = InferenceDataset(test_data, tokenizer, args.max_input_len)
        test_sampler = DistributedEvalSampler(test_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=2, sampler=test_sampler)

        all_decode_result = []
        all_labels = []
        count = 0
        with torch.no_grad():
            if label_list == []:
                if args.rank == 0:
                    test_dataloader = tqdm(test_dataloader, ncols=120, leave=False)
                for test_data in test_dataloader:
                    outputs = model.generate(
                        input_ids=test_data["input_ids"].to(args.device),
                        attention_mask=test_data["attention_mask"].to(args.device),
                        max_length=args.max_input_len + args.max_output_len,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    if outputs.size(1) < args.max_input_len + args.max_output_len:
                        batch_pred_padding = torch.ones((outputs.size(0), args.max_input_len + args.max_output_len - outputs.size(1)), dtype=outputs.dtype).cuda() * 2
                        outputs = torch.cat([outputs, batch_pred_padding], dim=1)

                    batch_out_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    batch_output = []
                    for idx, output in enumerate(batch_out_sentences):
                        output = output[len(test_data["input"][idx]):]
                        batch_output.append(output.strip())
                    all_decode_result.extend(batch_output)
                    all_labels.extend(test_data["label"])
                    if args.rank == 0:
                        count += len(batch_output) * args.world_size
            else:
                if args.rank == 0:
                    test_dataloader = tqdm(test_dataloader, ncols=120, leave=False)
                for test_data in test_dataloader:
                    input_text = test_data["input"]  # List[string] len=batch_size
                    gold_label = test_data["label"]  # List[string] len=batch_size

                    new_reqs = []
                    for i in input_text:
                        for l in label_list:
                            context_enc = tokenizer.encode(i)
                            continuation_enc = tokenizer.encode(i + l)[len(context_enc):]
                            key = (i, l)
                            value = (context_enc, continuation_enc)
                            new_reqs.append((key, value))

                    input_texts = [x[0][0] + x[0][1] for x in new_reqs]
                    inputs = tokenizer(
                        input_texts,
                        return_tensors='pt',
                        padding="longest",
                        max_length=args.max_input_len + args.max_output_len,
                        truncation=True,
                        return_attention_mask=True,
                    )

                    logits = model(inputs['input_ids'].to(model.device), attention_mask=inputs['attention_mask'].to(model.device))
                    logits = F.log_softmax(logits[0], dim=-1).cpu()

                    all_result = []
                    for one_req, one_logits in zip(new_reqs, logits):
                        key, value = one_req
                        _, cont_toks = value

                        # Slice to original seq length
                        contlen = len(cont_toks)
                        one_logits = one_logits[-contlen-1 : -1].unsqueeze(0)  # [1, seq, vocab]

                        # Check if per-token argmax is exactly equal to continuation
                        greedy_tokens = one_logits.argmax(dim=-1)
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                        if len(greedy_tokens[0]) != len(cont_toks[0]):
                            # 超出最大长度限制
                            answer = -float('inf')
                        else:
                            # Obtain log-probs at the corresponding continuation token indices
                            # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                            one_logits = torch.gather(one_logits,  dim=2, index=cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                            # Answer: (log prob, is-exact-match)
                            answer = float(one_logits.sum())
                        all_result.append(answer)
                    all_result = np.asarray(all_result)
                    all_result = all_result.reshape(-1, len(label_list))  # bsz, 2
                    gold_label_list = [x.split(", ") for x in gold_label]
                    gold_idx = [[label_list.index(x) for x in y] for y in gold_label_list]
                    all_result = np.argmax(all_result, axis=1)
                    
                    all_decode_result.extend(all_result)
                    all_labels.extend(gold_idx)
                    if args.rank == 0:
                        count += len(all_result) * args.world_size
                    
        dist.barrier()
        gather_data = [None for _ in range(args.world_size)]
        gather_label = [None for _ in range(args.world_size)]
        dist.all_gather_object(gather_data, all_decode_result)
        dist.all_gather_object(gather_label, all_labels)
        if args.rank == 0:
            preds = []
            labels = []
            for j in range(len(gather_data[0])):
                for i in range(len(gather_data)):
                    if j < len(gather_data[i]):
                        prediction = gather_data[i][j]
                        label = gather_label[i][j]
                        preds.append(prediction)
                        labels.append(label)
            if save_result:
                current_time = datetime.now()
                save_time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")
                with open(f"result/{save_time_string}.{task._cluster}_{task._name}.result.txt", "w", encoding="utf-8") as fw:
                    for p, l in zip(preds, labels):
                        fw.write(json.dumps({"pred": p, "label": l}) + "\n")
            result = task.compute_metrics(preds, labels)
            tqdm.write(f"{task._cluster}_{task._name}:\t{json.dumps(result)}")
            task_result.append((f"{task._cluster}_{task._name}", result))
            for k in result.keys():
                fw.write(f"{task._cluster}_{task._name}, {k}, {result[k]}\n")
                fw.flush()
    if args.rank == 0:
        fw.close()

if __name__ == '__main__':
    if args.gpus < 0:
        args.gpus = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='8889'
    args.test_batch_size = 2
    tasks = []
    label_lists = []
    
    # query description
    task = QueryDescription(name="gov2", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryDescription(name="trec_robust", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryDescription(name="trec_covid", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryDescription(name="fire", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])

    # query expansion
    task = QueryExpansion(name="gov2", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryExpansion(name="trec_robust", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryExpansion(name="trec_covid", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryExpansion(name="fire", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryExpansion(name="query2doc", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryExpansion(name="trec_cast", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryExpansion(name="trec_web", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # query reformulation
    task = QueryReformulation(name="codec", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryReformulation(name="qrecc", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryReformulation(name="canard", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryReformulation(name="trec_cast", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryReformulation(name="gecor", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # query clarification
    task = QueryClarification(name="mimics", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryClarification(name="mimics_duo", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryClarification(name="clariq_fkw", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = QueryClarification(name="raocq", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # query subtopic generation
    task = QuerySubtopicGeneration(name="trec_web", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # query suggestion
    task = QuerySuggestion(name="aol", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # query matching
    task = QueryMatching(name="msrp", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["yes", "no"])
    
    # query intent classification
    task = QueryIntentClassification(name="mantis", shot=args.n_shots, setting=args.setting, multi_label=True)
    tasks.append(task)
    label_lists.append(["original question", "further details", "other", "information request", "potential answer", "positive feedback", "negative feedback", "greetings / gratitude", "follow up question"])
    
    task = QueryIntentClassification(name="orcas_i", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["factual", "abstain", "instrumental", "transactional", "navigational"])
    task = QueryIntentClassification(name="trec_web", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["faceted", "ambiguous", "navigational", "informational"])
    
    # fact verification
    task = FactVerification(name="fever", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["support", "refute"])
    task = FactVerification(name="climate_fever", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["support", "refute", "disputed", "not enough information"])
    task = FactVerification(name="scifact", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["support", "refute"])
    
    # conversational qa
    task = ConversationalQA(name="coqa", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = ConversationalQA(name="quac", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # summarization
    task = Summarization(name="cnndm", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = Summarization(name="xsum", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = Summarization(name="wikisum", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = Summarization(name="multinews", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    # reading comprehension
    task = ReadingComprehension(name="squad", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = ReadingComprehension(name="hotpot_qa", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = ReadingComprehension(name="ms_marco", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = ReadingComprehension(name="boolq", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append(["true", "false"])
    task = ReadingComprehension(name="webglm_qa", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    task = ReadingComprehension(name="trivia_qa", shot=args.n_shots, setting=args.setting)
    tasks.append(task)
    label_lists.append([])
    
    assert len(tasks) == len(label_lists)
    mp.spawn(test_model, nprocs=args.gpus, args=(args, tasks, label_lists, False))
