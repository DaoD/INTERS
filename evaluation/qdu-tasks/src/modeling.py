import re
import torch
import copy
import time
import random
import numpy as np
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    logging
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Mapping, Tuple, List, Optional
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.get_logger(__name__)


@dataclass
class RerankResult:
    doc_id: int
    doc_score: int


class LM(nn.Module):
    def __init__(self, model_name_or_path, tokenizer_name_or_path, padding_side="left", dtype="bf16", device_map=None, use_flash_attention_2=False, access_token=None, cache_dir="/share/LMs", accelerator: Accelerator=None) -> None:
        super().__init__()

        logger.info(f"loading tokenizer from {tokenizer_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir, padding_side=padding_side, trust_remote_code=True)
        if tokenizer.pad_token is None:
            # NOTE: for models like Qwen, there is no pre-defined eos tokens
            if tokenizer.eos_token is None:
                pad_token = "<|endoftext|>"
            else:
                pad_token = tokenizer.eos_token
            tokenizer.pad_token = pad_token

        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        if device_map is None:
            if accelerator is not None:
                device_map = {"": accelerator.device}
            else:
                device_map = {"": "cpu"}

        logger.info(f"loading model from {model_name_or_path}...")
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir, 
                torch_dtype=dtype, 
                trust_remote_code=True, 
                device_map=device_map,
                token=access_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir, 
                torch_dtype=dtype, 
                trust_remote_code=True, 
                device_map=device_map,
                use_flash_attention_2=use_flash_attention_2,
                token=access_token,
            )

        self.config = model.config
        self.tokenizer = tokenizer

        # move model to the correct device / enable mixed precision
        if accelerator is not None:
            self.model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)
        else:
            self.model = model

        self.rng = np.random.default_rng(42)
        # ignore dropout and gradients
        self.eval()

    @property
    def device(self):
        return self.model.device
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
    
    def _reorder_cache(self, *args, **kwargs):
        return self.model._reorder_cache(*args, **kwargs)

    def _move_to_device(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._move_to_device(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._move_to_device(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        else:
            return data

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, return_new_tokens_only=True, decode=True, accelerator:Optional[Accelerator]=None, **inputs):
        outputs = self.model.generate(**inputs)

        if return_new_tokens_only:
            if self.model.config.is_encoder_decoder:
                if "decoder_input_ids" in inputs:
                    start_idx = inputs["decoder_input_ids"].shape[1] + 1
                else:
                    start_idx = 1
            else:
                start_idx = inputs["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

        if accelerator is not None:
            # must be contiguous
            outputs = outputs.contiguous()
            outputs = accelerator.pad_across_processes(outputs, pad_index=self.tokenizer.pad_token_id, dim=1)
            outputs = accelerator.gather_for_metrics(outputs)
        
        outputs = outputs.tolist()
        if decode:
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    @torch.no_grad()
    def rerank_pointwise(self, dataloader, accelerator):
        """
        Args:
            dataloader: return a batch of input_ids and attention_mask, each of which is a query-doc pair.
        """
        if accelerator is not None and type(dataloader) is DataLoader:
            dataloader = accelerator.prepare(dataloader)

        is_encoder_decoder = self.model.config.is_encoder_decoder
        yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        rerank_results = defaultdict(list)
        for i, x in enumerate(tqdm(dataloader, desc="Pointwise Reranking", ncols=120)):
            query_ids = x.pop("query_id")
            doc_ids = x.pop("doc_id")

            if is_encoder_decoder:
                raise NotImplementedError

            else:
                logits = self.model(**x).logits[:, -1]          # batch_size, vocab_size

            yes_and_no_logits = logits[:, [yes_id, no_id]]      # batch_size, 2
            # NOTE: normalize so that different documents are comparable
            yes_and_no_logits = torch.softmax(yes_and_no_logits, dim=1)
            doc_scores = yes_and_no_logits[:, 0]                    # batch_size

            # gather outputs across devices
            if accelerator is not None:
                query_ids = accelerator.gather_for_metrics(query_ids)
                doc_ids = accelerator.gather_for_metrics(doc_ids)
                doc_scores = accelerator.gather_for_metrics(doc_scores)
            
            for query_id, doc_id, doc_score in zip(query_ids.tolist(), doc_ids.tolist(), doc_scores.tolist()):
                rerank_result = RerankResult(doc_id=doc_id, doc_score=doc_score)
                rerank_results[query_id].append(rerank_result)

        # sort candidates of each query
        for query_id, res in rerank_results.items():
            sorted_res = sorted(res, key=lambda x: x.doc_score, reverse=True)
            rerank_results[query_id] = sorted_res

        return dict(rerank_results)

    def compare(self, query: str, docs: List, prompt_template: str, fewshot_prompt: Optional[str]=None):
        doc1, doc2 = docs[0], docs[1]
        #doc1是第ind个，doc2是第ind-1个
        input_texts = [prompt_template.format(query=query, doc1=doc1, doc2=doc2) + "[1]", prompt_template.format(query=query, doc1=doc1, doc2=doc2) + "[2]", 
                       prompt_template.format(query=query, doc1=doc2, doc2=doc1) + "[1]", prompt_template.format(query=query, doc1=doc2, doc2=doc1) + "[2]"]#构成两条prompt

        # NOTE: add fewshot prompt
        if fewshot_prompt is not None:
            input_texts = [fewshot_prompt + x for x in input_texts]

        # 需要将prompt和标签拼在一起，以计算输出概率
        # 第一个：认为原序靠后的文档比靠前的文档更相关，换位置
        # 第二个：认为原序靠后的文档不如靠前的文档相关，不换位置
        # 第三个：认为原序靠后的文档不如靠前的文档相关，不换位置
        # 第四个：认为原序靠后的文档比靠前的文档更相关，换位置
        inputs = self.tokenizer(input_texts, return_tensors="pt")#得到prompt的input
        #print("input: ", inputs.device())
        
        target_texts = ["[1]", "[2]", "[1]", "[2]"]
        targets = self.tokenizer(target_texts, add_special_tokens=False)["input_ids"]#获得目标输出的id
        targets_length = [len(x) for x in targets]#获得目标输出的id长度，即每个输出对应几个id
        
        labels = inputs["input_ids"].clone()
        for i, label in enumerate(labels):
            labels[i, :-targets_length[i]] = -100#将需要计算的id保留，前面原prompt对应的id标记为-100，在模型计算时会省略掉
        # inputs["labels"] = labels
        inputs = inputs.to(self.device)

        outputs = self.model(**inputs)#.to(self.device)
        logits = outputs["logits"].detach()#获得输出的logits值
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        labels = labels.to(logits.device)
        batch_size = logits.shape[0]#获得batch_size
        token_loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            labels.reshape(-1),
            reduction="none"
        ).reshape(batch_size, -1)#获得损失，在这里，损失是-log(p(t=n|t<n))
        
        valid_token_num = (labels != -100).sum(-1)#每条数据的有效token个数（非-100的）
        #将损失加起来。由于是对数损失，因此相当于内部概率乘积，得到最终输出后几位（【1】【2】等）的概率
        token_prob = - token_loss.sum(-1) / valid_token_num

        # for x in inputs["input_ids"]:
        #     print("-"*50)
        #     print(self.tokenizer.decode(x))
        # # labels2 = labels.clone()
        # # labels2[labels2==-100] = self.tokenizer.pad_token_id
        # # print(f"labels:                 {self.tokenizer.batch_decode(labels2)}")
        # print(f"likelihood:             {token_prob}")
        # print(f"switch?                 {token_prob[0] > token_prob[1] and token_prob[2] < token_prob[3]}")
        # input()

        if token_prob[0] > token_prob[1] and token_prob[2] < token_prob[3]:
            return f'change'
        return f'not change'
        
    @torch.no_grad()
    def rerank_pairwise(self, dataloader, prompt_template, accelerator):
        rerank_results = defaultdict(list)
        if accelerator is not None and type(dataloader) is DataLoader:
            dataloader = accelerator.prepare(dataloader)
        
        for _, ranking_result in enumerate(tqdm(dataloader, desc="Pairwise Reranking", ncols=120)):
            #采用bubblesort方法
            k = ranking_result["doc_ids"].size(dim=1)
            last_end = k - 1
            query = ranking_result["query"]
            batch_size = ranking_result["doc_ids"].size(dim=0)

            ranking_result["doc_ids"] = ranking_result["doc_ids"].tolist()

            fewshot_prompts = ranking_result["fewshot_prompt"]

            for i in range(batch_size):#对一个batch中的每个数据分别处理
                # NOTE: convert doc_ids to list
                pairs = list(zip(ranking_result["docs"][i], ranking_result["doc_ids"][i]))
                self.rng.shuffle(pairs)

                shuffled_docs, shuffled_doc_ids = zip(*pairs)
                shuffled_docs = list(shuffled_docs)
                shuffled_doc_ids = list(shuffled_doc_ids)
                ranking_result["docs"][i] = shuffled_docs
                ranking_result["doc_ids"][i] = shuffled_doc_ids

                if fewshot_prompts is not None:
                    fewshot_prompt = fewshot_prompts[i]
                else:
                    fewshot_prompt = None
                
                for j in range(k):
                    current_ind = last_end
                    is_change = False
                    while True:
                        if current_ind <= j:
                            break
                        doc1 = ranking_result["docs"][i][current_ind]
                        doc2 = ranking_result["docs"][i][current_ind - 1]
                        output = self.compare(query[i], [doc1, doc2], prompt_template=prompt_template, fewshot_prompt=fewshot_prompt)#输入模型比较两个documents的顺序
                        if output == 'change':#靠后的document比靠前的document更相关,则交换位置（该步将两个documents交换顺序后又输入了一遍）
                            ranking_result["docs"][i][current_ind - 1], ranking_result["docs"][i][current_ind] = ranking_result["docs"][i][current_ind], ranking_result["docs"][i][current_ind - 1]
                            ranking_result["doc_ids"][i][current_ind - 1], ranking_result["doc_ids"][i][current_ind] = ranking_result["doc_ids"][i][current_ind], ranking_result["doc_ids"][i][current_ind - 1]
                            """
                            比如输入doc1 = ranking[current_ind],doc2 = ranking[current_ind - 1]
                            如果当document A为doc1,documen B为doc2时,模型输出A更相关,并且当document A为doc2,documen B为doc1时,模型输出B更相关
                            则认为doc1比doc2更相关,需要互换位置
                            """

                            if not is_change:
                                is_change = True
                                if last_end != k - 1:  # skip unchanged pairs at the bottom
                                    last_end += 1
                        if not is_change:
                            last_end -= 1
                        current_ind -= 1
            query_ids = ranking_result.pop("query_id")
            doc_ids = torch.tensor(ranking_result.pop("doc_ids"), device=self.device)
            if accelerator is not None:
                query_ids = accelerator.gather_for_metrics(query_ids)
                doc_ids = accelerator.gather_for_metrics(doc_ids)
            for query_id, doc_id in zip(query_ids.tolist(), doc_ids.tolist()):
                for doc_id_i in doc_id:
                    ranking_result = RerankResult(doc_id=doc_id_i, doc_score=0)
                    rerank_results[query_id].append(ranking_result)
        return dict(rerank_results)
    
    def permutation_pipeline(self, item=None, rank_start=0, rank_end=100, prompt_template=None):
        f_query = item["query"]
        num = len(item['hits'][rank_start: rank_end])
        fewshot_prompt = item["fewshot_prompt"]
        
        rank = 0
        docs = ""
        for hit in item['hits'][rank_start: rank_end]:
            rank += 1
            if isinstance(hit['document'], str):
                document = hit['document'].strip()
            else:
                print(item['hits'])
                raise ValueError("document should be a string")
            docs += f"[{rank}] " + document + "\n"
        messages = prompt_template.format(query=f_query, num=num, docs=docs)
        # messages = prompt_template.format(query=query, num=rank, docs=docs)
        if fewshot_prompt is not None:
            messages = fewshot_prompt + messages

        input_ids = self.tokenizer(messages, return_tensors="pt", padding='longest', truncation=False).input_ids.to(self.model.device)
        output_ids = self.model.generate(input_ids,
                                        do_sample=False,
                                        temperature=0.0,
                                        top_p=None,
                                        max_new_tokens=500,
                                        pad_token_id=self.tokenizer.eos_token_id,)
        permutation = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
        
        response = [int(match.group(1)) -1 for match in re.finditer(r"(\d+)", permutation)]
        response = [x for x in response if 0<= x < 10]
        if len(response) == 0:
            return item

        new_response = []
        for x in response:
            if x not in new_response:
                new_response.append(x)
        response = new_response

        if len(response) < 10:
            full_set = set(range(0, 10))
            missing_number = full_set - set(response)
            response.extend(list(missing_number))

        candidates = copy.deepcopy(item['hits'][rank_start: rank_end])
        rerank_candidates = [candidates[x] for x in response]
        item['hits'][rank_start: rank_end] = rerank_candidates

        # if torch.distributed.get_rank() == 0:
        #     print(messages)
        # print(f"outputs:                    {response}")
        # input()
        return item

    def sliding_windows(self, item=None, prompt_template=None, 
                        rank_start=0, rank_end=100, window_size=5, step=3):
        item = copy.deepcopy(item)
        end_pos = rank_end
        start_pos = rank_end - window_size
        while start_pos >= rank_start:
            # if torch.distributed.get_rank() == 0:
            #     print(start_pos, rank_start)
            start_pos = max(start_pos, rank_start)
            item = self.permutation_pipeline(item, start_pos, end_pos, prompt_template)
            end_pos = end_pos - step
            start_pos = start_pos - step
        return item
    
    @torch.no_grad()
    def rerank_listwise(self, dataloader, prompt_template, accelerator, window, stride):
        rerank_results = defaultdict(list)
        if accelerator is not None and type(dataloader) is DataLoader:
            dataloader = accelerator.prepare(dataloader)

        for _, ranking_data in enumerate(tqdm(dataloader, desc="Listwise Reranking", ncols=120)):
            batch_size = ranking_data["doc_ids"].size(dim=0)
            items = [{} for _ in range(batch_size)]
            doc_start = [0] * batch_size
            doc_end = []
            for i in range(batch_size):
                items[i]["query"] = ranking_data["query"][i]
                items[i]["query_id"] = ranking_data["query_id"][i].item()
                items[i]["fewshot_prompt"] = ranking_data["fewshot_prompt"][i]
                items[i]["hits"] = [{"document": doc, "docid": docid.item()} 
                                    for doc, docid in zip(ranking_data["docs"][i], ranking_data["doc_ids"][i])]
                self.rng.shuffle(items[i]["hits"])
                doc_end.append(len(items[i]["hits"]))
            prompt_templates = [prompt_template] * batch_size
            windows = [window] * batch_size
            strides = [stride] * batch_size
            items = list(map(self.sliding_windows, items, prompt_templates, doc_start, doc_end, windows, strides))
            """
            items返回类型如下
            item = {
                'query': query(string),
                'query_id': qid(int),
                'hits': [
                    {'document': document(string), 'docid': docid(int)},
                    {'document': document(string), 'docid': docid(int)},
                    {'document': document(string), 'docid': docid(int)},
                ]
            }
            现在要将其转成这种格式:
            {
                qid: list[RerankResult],
                ......
            }
            """
            query_ids = ranking_data.pop("query_id")
            doc_ids = torch.tensor([[hit["docid"] for hit in item["hits"]] for item in items], device=self.device)

            # print(doc_ids)
            # input()

            if accelerator is not None:
                query_ids = accelerator.gather_for_metrics(query_ids)
                doc_ids = accelerator.gather_for_metrics(doc_ids)
            for query_id, doc_id in zip(query_ids.tolist(), doc_ids.tolist()):
                for doc_id_i in doc_id:
                    ranking_result = RerankResult(doc_id=doc_id_i, doc_score=0)
                    rerank_results[query_id].append(ranking_result)

        return dict(rerank_results)


def get_model_and_tokenizer(model_args, accelerator=None, **kwargs):
    """Load model and tokenizer. Possibly load LoRA for the model."""

    from .args import ModelArgs
    model_args: ModelArgs

    model_args = asdict(model_args)
    model_args.update(**kwargs)
    
    model = LM(
        model_name_or_path=model_args["model_name_or_path"],
        tokenizer_name_or_path=model_args["tokenizer_name_or_path"],
        padding_side=model_args["padding_side"],
        dtype=model_args["dtype"],
        cache_dir=model_args["model_cache_dir"],
        device_map=model_args["device_map"],
        use_flash_attention_2=model_args["use_flash_attention_2"],
        access_token=model_args["access_token"],
        accelerator=accelerator
    )

    # load lora
    if model_args["lora"] is not None:
        from peft import PeftModel
        logger.info(f"loading lora from {model_args['lora']}...")
        model = PeftModel.from_pretrained(model, model_args["lora"])
        model = model.merge_and_unload()
            
    return model, model.tokenizer