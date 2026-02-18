import os
import datasets
import numpy as np
from typing import List
from dataclasses import dataclass, field, asdict
from torch.utils.data import DataLoader
from collections import defaultdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging

from src import ModelArgs, Metric, DatasetProcessFn, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs


logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default=None,
        metadata={'help': 'The evaluation json data path.'}
    )
    fewshot_data: str = field(
        default=None,
        metadata={'help': 'The fewshot json data path.'}
    )
    output_dir: str = field(
        default="data/results/rerank",
        metadata={'help': 'Output directory for results and logs.'}
    )
    batch_size: int = field(
        default=16,
        metadata={'help': 'Evaluation batch size.'}
    )

    query_max_length: int = field(
        default=64,
        metadata={'help': 'How many tokens at maximum?'}
    )
    doc_max_length: int = field(
        default=512,
        metadata={'help': 'How many tokens at maximum?'}
    )

    with_description: bool = field(
        default=True,
        metadata={'help': "Whether to add task description"}
    )
    rerank_method: str = field(
        default="pointwise",
        metadata={'help': 'How to evaluate reranking? {pointwise, pairwise, listwise, no}'}
    )
    dataset_name: str = field(
        default="msmarco",
        metadata={'help': 'Select one from [msmarco | trec_covid | nfcorpus | nq | fiqa | hotpot_qa | arguana | touche | cqadupstack | quora | dbpdia | scidocs | climate_fever | fever | scifact]'},
    )
    hits: int = field(
        default=100,
        metadata={'help': 'How many candidates to rerank?'}
    )
    shots: int = field(
        default=None,
        metadata={'help': 'How many shots to use for fewshot testing?'}
    )

    metrics: List[str] = field(
        default_factory=lambda: ["mrr", "recall", "ndcg"],
        metadata={'help': 'List of metrics. {rouge, acc}'}
    )
    cutoffs: List[int] = field(
        default_factory=lambda: [1, 5, 10],
        metadata={'help': 'Cutoffs to evaluate retrieval metrics.'}
    )
    listwise_window: int = field(
        default=10,
        metadata={'help': 'How long is the window in listwise?'}
    )
    listwise_stride: int = field(
        default=5,
        metadata={'help': 'How long is the step in listwise?'}
    )


TASK_DESCRIPTION = "In the reranking task, search engines must understand the relationship between the user's query, which may be keywords or a sentence, and the potential documents. The goal is to ensure that the most relevant documents, those that best cover the user's information needs, are ranked highest. This requires a nuanced understanding of both the query's intent and the content of the documents."

PROMPT_TEMPLATE = {
    "pointwise": {
        "msmarco": "Assess the relevance between the provided document:\n{text}\nand the query: \"{query}\". Respond with 'Yes' if the document is relevant to the query or 'No' if not.",
        "trec_covid": "Document: {text}\n\nQuery: {query}\n\nAssess the relevance of the provided document in the context of the COVID-19-related query. Answer 'Yes' if the document explicitly addresses or pertains to the query, or 'No' if it is unrelated.",
        "nfcorpus": "Assess the relevance of the medical document:\n{text}\nin relation to the search query \"{query}\". Determine if the document is relevant by responding with 'Yes' for relevance or 'No' for irrelevance.",
        "nq": "Review the content of the document:\n{text}\nand ascertain its relevance to the topic: \"{query}\". Provide your determination by responding with either 'Yes' for relevance or 'No' for irrelevance.",
        "fiqa": "Evaluate the financial document:\n{text}\nin the context of the query: \"{query}\". Determine if the document is relevant to the query and provide your judgment with 'Yes' for relevance or 'No' for irrelevance.",
        "hotpot_qa": "Analyze the relationship between the document:\n{text}\nand the query: \"{query}\". Offer a definitive response by indicating whether they are relevant, and reply with either 'Yes' for relevance or 'No' for irrelevance.",
        "arguana": "Document:\n\n{text}\n\nQuery:\n\n{query}\n\nDetermine their relevance and return the judgment about the document's relevance to the query by responding with either 'Yes' for relevance or 'No' for irrelevance.",
        "touche": "Evaluate the relevance between the given document on controversial subjects:\n{text}\nand the query: \"{query}\", and provide a clear 'Yes' for relevance or 'No' for irrelevance judgment regarding their relevance.",
        "cqadupstack": "Evaluate the relevance between the given document on controversial subjects:\n{text}\nand the query: \"{query}\", and provide a clear 'Yes' for relevance or 'No' for irrelevance judgment regarding their relevance.",
        "quora": "Determine the relevance between the document:\n{text}\nand the query: {query}. Judge whether they are related by responding with 'Yes' for relevance or 'No' for irrelevance.",
        "dbpedia": "Determine the relevance of the document:\n{text}\nto the query: \"{query}\". Conclude with 'Yes' for relevance or 'No' for irrelevance.",
        "scidocs": "Determine the correlation between the provided literature-related document:\n{text}\nand the query: \"{query}\". Conclude if they are closely connected with a 'Yes' for relevance or 'No' for irrelevance.",
        "climate_fever": "Analyze the correlation between this document on the topic of climate change:\n{text}\nand the following query: {query}. Determine if the document is relevant to the query. Respond with 'Yes' for relevance or 'No' for irrelevance.",
        "fever": "Assess the relationship between the given document:\n{text}\nand the query: \"{query}\" to determine if the document is relevant. Answer with 'Yes' for relevance or 'No' for irrelevance.",
        "scifact": "Assess the relevance between the scientific document:\n{text}\nand query: \"{query}\", and state 'Yes' or 'No' to reflect the relevance judgment. Answer 'Yes' for relevance and 'No' for irrelevance."
    },
    "pairwise": {
        "msmarco": "Consider a query \"{query}\" alongside two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nDecide which document is more relevant to the given query by providing the corresponding document identifier.",
        "trec_covid": "Given a query: \"{query}\" and two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nEach marked with a unique identifier and related to COVID-19, assess and identify which document is more closely related to the query. Respond with the identifier of the more relevant document.",
        "nfcorpus": "Assess which of the two medical field documents is more relevant to the provided query \"{query}\". Each document is presented with a unique identifier.\nDocuments:\n\n[1] {doc1}\n\n[2] {doc2}\n\nIdentify and return the identifier of the document that best aligns with the query.",
        "nq": "Evaluate the relevance of the provided query \"{query}\" to a pair of documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\neach identified separately. Determine which document is more relevant to the query and return the identifier of the more relevant document.",
        "fiqa": "Evaluate the relevance of the query: \"{query}\" and the pair of financial documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\neach assigned a unique identifier. Determine which document is more relevant to the provided query and specify its document identifier.",
        "hotpot_qa": "Compare the relevance of two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nto the provided query: \"{query}\". Identify the document with the higher relevance by specifying its identifier.",
        "arguana": "Evaluate the relevance of two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nto the provided query \"{query}\". Express the identifier of the document that is more relevant to the query.",
        "touche": "Given a query: \"{query}\" and two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\neach with its unique identifier, determine which document is more relevant to the given query by providing the document identifier.",
        "cqadupstack": "Compare the relevance of two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nto the query: \"{query}\", and specify the document identifier that has higher relevance.",
        "quora": "With a given query: \"{query}\" and two unique documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nIdentify the document that aligns more closely with the query by stating its identifier.",
        "dbpedia": "Given a query: \"{query}\" and two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\neach identified by a distinct number. Determine the document identifier for the one more relevant to the provided query.",
        "scidocs": "Given a query:\"{query}\" and two literature-related documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\neach with a unique identifier, identify which document is more relevant to the query by specifying its identifier.",
        "climate_fever": "Given this query:\"{query}\" and two climate change documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\neach with a unique identifier, identify which document aligns more closely with the query by indicating the document's identifier.",
        "fever": "Evaluate two documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nagainst a given query: \"{query}\" and identify the document that is more relevant by its identifier.",
        "scifact": "Analyze the relevance of two scientific documents:\n\n[1] {doc1}\n\n[2] {doc2}\n\nto the query: \"{query}\" and identify the more relevant document by its identifier."
    },
    "listwise": {
        "msmarco": "Here are {num} documents:\n{docs}\nand a query \"{query}\", and each document is indicated by a number identifier. Please sort the documents in an order based on their relevance to the above query by returning an identifier list. Be careful to sort documents in order of their relevance to the query from highest to lowest. Result: ",
        "trec_covid": "Given {num} documents:\n{docs}\neach pertaining to COVID-19, and a specific query: \"{query}\", organize the documents in order of relevance to the query. List the identifiers of these documents starting from the most relevant to the least relevant. Result: ",
        "nfcorpus": "Arrange the given {num} medical field documents:\n{docs}\nin order of relevance to the specified query \"{query}\", with the most relevant document at the top and the least relevant at the bottom. Use their unique number identifiers to indicate the sequence of relevance. Result: ",
        "nq": "Arrange the provided {num} documents:\n{docs}\nin accordance with their relevance to the specified query \"{query}\". Utilize number identifiers to denote the sequence of relevance, with the most relevant document at the top and the least relevant document at the end. Result: ",
        "fiqa": "Assess the relevance of the query: \"{query}\" to a set of {num} financial documents:\n{docs}\nGenerate a list of document identifiers, arranging them from the most relevant to the least relevant in relation to the query, and return the identifiers list. Result: ",
        "hotpot_qa": "Rank the following {num} documents in descending order of relevance to the query: \"{query}\"\n{docs}\nProvide the list of identifiers. Result: ",
        "arguana": "Rank the {num} documents:\n{docs}\n in descending order of relevance to the provided query: \"{query}\". Return the list of identifiers associated with each document in order. Result:",
        "touche": "Rerank the provided {num} documents on a controversial topic:\n{docs}\nbased on their relevance to the query \"{query}\". Return a list of identifiers in the order of descending relevance. Result: ",
        "cqadupstack": "Rerank the provided {num} documents on a controversial topic:\n{docs}\nbased on their relevance to the query \"{query}\". Return a list of identifiers in the order of descending relevance. Result: ",
        "quora": "Rank {num} documents:\n{docs}\nby their relevance to the query: \"{query}\" in descending order. List their identifiers. Result: ",
        "dbpedia": "Rank {num} documents:\n{docs}\nbased on their relevance to the specified query \"{query}\". Return the list of identifiers in descending order. Result: ",
        "scidocs": "Rank {num} documents:\n{docs}\nin order of their relevance to the query: {query}. List the identifiers starting with the most relevant document. Result: ",
        "climate_fever": "Rerank these {num} climate change documents:\n{docs}\nby their relevance to the query: \"{query}\" and list the identifiers. Rank from most to least relevant. Result: ",
        "fever": "Rank a set of {num} documents:\n{docs}\nby their relevance to a query: \"{query}\". List their identifiers in order of decreasing relevance. Result: ",
        "scifact": "Order the provided {num} scientific documents:\n{docs}\n based on their relevance to the query: \"{query}\", from most to least relevant. Only output a list of identifiers. Result: "
        # "Here is a query:\n\n{query}\n\nHere are {num} documents, and each document has its own identifier:\n\n{docs}\n\nRank the {num} documents above based on their relevance to the given query. The documents should be listed in descending order using identifiers. The most relevant documents should be listed first. The output format should be [] > [], e.g., [1] > [2]. Each two identifiers are separated by ' > '. Only response the ranking results, do not say any word or explain.",
    },
    "no": defaultdict(lambda: ""),
}


def truncate(text, tokenizer, max_length):
    if tokenizer is not None:
        return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True))
    else:
        return text


def process_rerank(tokenizer, rerank_method, prompt_template, query_max_length=64, doc_max_length=512, hits=10, fewshot_data=None, shots=None, cache_dir=None):
    if fewshot_data is not None:
        fewshot_data = datasets.load_dataset("json", data_files=fewshot_data, split="train", cache_dir=cache_dir)
        rng = np.random.default_rng(42)
        indices = rng.choice(range(len(fewshot_data)), size=shots, replace=False).tolist()

        fewshot_prompt = []

        for index in indices:
            item = fewshot_data[index]
            # NOTE: do not use query, pos, neg here, because they are the arguments for the _process function
            fs_query = item["query"]
            fs_pos = item["pos"]
            fs_neg = item["neg"]

            fs_query = truncate(fs_query, tokenizer, query_max_length)

            if rerank_method == "pointwise":
                # sample 1 candidate from the union of pos and neg
                candidate = fs_pos + fs_neg
                candidate_range = range(len(candidate))
                candidate_index = rng.choice(candidate_range).tolist()
                candidate = truncate(candidate[candidate_index], tokenizer, doc_max_length)
                prompt = prompt_template.format(query=fs_query, text=candidate)

                if candidate_index < len(fs_pos):
                    # sampled from pos
                    prompt += " Yes."
                else:
                    prompt += " No."

            elif rerank_method == "pairwise":
                # sample 1 positive and 1 negative for comparison
                fs_pos = rng.choice(fs_pos).tolist()
                fs_neg = rng.choice(fs_neg).tolist()

                reverse = rng.choice([True, False])
                if reverse:
                    prompt = prompt_template.format(query=fs_query, doc1=fs_neg, doc2=fs_pos)
                    prompt += "[2]"
                else:
                    prompt = prompt_template.format(query=fs_query, doc1=fs_pos, doc2=fs_neg)
                    prompt += "[1]"
                
            elif rerank_method == "listwise":
                len_pos = len(fs_pos)
                len_neg = len(fs_neg)
                len_all = len_pos + len_neg
                all_documents = fs_pos + fs_neg
                result = list(range(1, len_all + 1))
                rng.shuffle(result)
                idx_document = {idx: document for idx, document in zip(result, all_documents)}
                idx_document = dict(sorted(idx_document.items()))
                result = [f"[{num}]" for num in result]
                result = " > ".join(result)
                docs = ""
                rank = 0
                for value in idx_document.values():
                    rank += 1
                    docs += f"[{rank}] " + value + "\n"
                prompt = prompt_template.format(query=fs_query, num=rank, docs=docs)
                prompt += " " + result
                
            else:
                raise NotImplementedError(f"Rerank method {rerank_method} not implemented for fewshot!")

            fewshot_prompt.append(prompt)
        
        fewshot_prompt = "\n\n".join(fewshot_prompt) + "\n\n"

    else:
        fewshot_prompt = None

    
    @DatasetProcessFn(augment=True)
    def _process_pointwise(query, pos=None, neg=None, pos_index=None, neg_index=None, pos_score=None, key=None, key_index=None, query_id=None, _index=None, **kwds):
        outputs = defaultdict(list)
        if pos_score is None:
            pos_score = [1 for _ in pos]

        # rerank positive and negative documents when there are no pre-defined candidates
        if neg is not None:
            key = pos + neg
            key_index = pos_index + neg_index

        if query_id is None:
            assert _index is not None, f"Make sure to set with_indices=True when there is no given query_id!"
            query_id = _index

        # truncate query
        query = truncate(query, tokenizer, query_max_length)
        # only rerank the top-hits candidates
        key = key[:hits]
        key_index = key_index[:hits]

        for doc_id, doc_text in zip(key_index, key):
            # truncate document
            doc_text = truncate(doc_text, tokenizer, doc_max_length)

            prompt = prompt_template.format(query=query, text=doc_text)
            # NOTE: prepend fewshot prompt
            if fewshot_prompt is not None:
                prompt = fewshot_prompt + prompt

            output = tokenizer(prompt)

            output["query_id"] = query_id
            output["doc_id"] = doc_id

            for k, v in output.items():
                outputs[k].append(v)
        return outputs


    @DatasetProcessFn()
    def _process_other(query, pos=None, neg=None, pos_index=None, neg_index=None, pos_score=None, key=None, key_index=None, query_id=None, _index=None, **kwds):
        if pos_score is None:
            pos_score = [1 for _ in pos]

        # rerank positive and negative documents when there are no pre-defined candidates
        if neg is not None:
            key = pos + neg
            key_index = pos_index + neg_index

        if query_id is None:
            assert _index is not None, f"Make sure to set with_indices=True when there is no given query_id!"
            query_id = _index
        
        # truncate query
        query = truncate(query, tokenizer, query_max_length)

        if len(key) < hits:
            return None
        
        # only rerank the top-hits candidates
        key = key[:hits]
        key_index = key_index[:hits]
        for i, k in enumerate(key):
            key[i] = truncate(k, tokenizer, doc_max_length)

        outputs = {
            "query": query,
            "query_id": query_id,
            "docs": key,
            "doc_ids": key_index
        }

        # always add fewshot prompt
        outputs["fewshot_prompt"] = fewshot_prompt
        return outputs

    if rerank_method == "pointwise":
        return _process_pointwise
    else:
        return _process_other


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    # NOTE: if skip, we just use CPU to load model
    # import json
    # with open("/share/peitian/Data/Datasets/searchgpt/qrels.train.pt.neg.do-not-overwrite.fewshot.jsonl", 'r', encoding='utf-8') as fr:
    #     lines = [json.loads(line.strip()) for line in fr.readlines()[:100]]
    # with open("/share/yutao/yifei/data/test_fewshot_100.jsonl", 'w', encoding='utf-8') as fw: 
    #     for line in lines:
    #         json.dump(line, fw)
    #         fw.write('\n')
    # with open("/share/peitian/Data/Datasets/searchgpt/qrels.dev.pt.key.do-not-overwrite.jsonl", 'r', encoding='utf-8') as fr:
    #     lines = [json.loads(line.strip()) for line in fr.readlines()[:100]]
    # with open("/share/yutao/yifei/data/test_100.jsonl", 'w', encoding='utf-8') as fw: 
    #     for line in lines:
    #         json.dump(line, fw)
    #         fw.write('\n')
    # return
        
    accelerator = Accelerator(cpu=args.cpu or args.rerank_method == "no")

    if args.rerank_method == "no":
        model = None
        tokenizer = None
        logger.info(f"directly evaluating {args.eval_data}...")
    else:
        model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)

    with accelerator.main_process_first():
        prompt_template = PROMPT_TEMPLATE[args.rerank_method][args.dataset_name]
        if args.with_description:
            prompt_template = TASK_DESCRIPTION + " " + prompt_template
        process_fn = process_rerank(
            tokenizer=tokenizer, 
            rerank_method=args.rerank_method, 
            prompt_template=prompt_template,
            query_max_length=args.query_max_length, 
            doc_max_length=args.doc_max_length,
            hits=args.hits,
            fewshot_data=args.fewshot_data,
            shots=args.shots,
            cache_dir=args.dataset_cache_dir,
        )
        dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, with_indices=True)

    if args.rerank_method == "no":
        # directly compose rerank results from retrieval results
        from src import RerankResult
        results = {}
        for x in dataset:
            query_id = x["query_id"]
            results[query_id] = [RerankResult(doc_id, 0) for doc_id in x["doc_ids"]]

    else:
        data_collator = DefaultDataCollator(tokenizer=tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            collate_fn=data_collator,
            pin_memory=model.device.index is not None,
        )
        
        if args.rerank_method == "pointwise":
            results = model.rerank_pointwise(dataloader, accelerator=accelerator)
        elif args.rerank_method == "pairwise":
            results = model.rerank_pairwise(dataloader, prompt_template=prompt_template, accelerator=accelerator)
        elif args.rerank_method == "listwise":
            results = model.rerank_listwise(dataloader, prompt_template=prompt_template, accelerator=accelerator, window=args.listwise_window, stride=args.listwise_stride)
        else:
            raise NotImplementedError(f"Rerank method {args.rerank_method} not implemented!")

    if accelerator.process_index == 0:
        result_path = Metric._get_save_path(args.eval_data, args.output_dir)
        Metric._save_rerank_result(results, result_path, eval_data=args.eval_data)
        metrics = Metric.get_metric_fn(metric_names=args.metrics, eval_data=args.eval_data, cutoffs=args.cutoffs)(results)

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))


if __name__ == "__main__":
    main()
