<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/logo1.jpg" width="150px">
</div>


## INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning</h2>
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p>

**Authors**: Yutao Zhu, Peitian Zhang, Chenghao Zhang, Yifei Chen, Binyu Xie, Zhicheng Dou, Zheng Liu, and Ji-Rong Wen

<p>
📃 <a href="https://arxiv.org/abs/2401.06532">ArXiv Paper</a>
  •
📚 <a href="https://huggingface.co/datasets/yutaozhu94/INTERS">Dataset</a>
</p>
<p>
🤗 HuggingFace Model List
</p>

| Model                                                                            | Backbone Model                                                          |
|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------|
| [INTERS-LLaMA-7b-Chat](https://huggingface.co/yutaozhu94/INTERS-LLaMA-7b-chat)   | [LLaMA-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| [INTERS-LLaMA-7b-Base](https://huggingface.co/yutaozhu94/INTERS-LLaMA-7b-base)   | [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)           |
| [INTERS-Mistral-7b](https://huggingface.co/yutaozhu94/INTERS-Mistral-7b)         | [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1)          |
| [INTERS-Minima-3b](https://huggingface.co/yutaozhu94/INTERS-Minima-3b)           | [Minima-2-3b](https://huggingface.co/GeneZC/MiniMA-2-3B)                |
| [INTERS-Falcon-1b](https://huggingface.co/yutaozhu94/INTERS-Falcon-1b)           | [Falcon-rw-1b](https://huggingface.co/tiiuae/falcon-rw-1b)              |

## News
- May, 2024: We are happy that INTERS has been accepted by ACL 2024 main conference!
- Feb, 2024: We have released the dataset, instruction templates, fine-tuned models, and evaluation scripts.

## Introduction
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/intro.jpg"  width="600px">
</div>

Large language models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. Despite this, their application to information retrieval (IR) tasks is still challenging due to the infrequent occurrence of many IR-specific concepts in natural language. While prompt-based methods can provide task descriptions to LLMs, they often fall short in facilitating a comprehensive understanding and execution of IR tasks, thereby limiting LLMs' applicability. To address this gap, in this work, we explore the potential of instruction tuning to enhance LLMs' proficiency in IR tasks. We introduce a novel instruction tuning dataset, INTERS, encompassing 20 tasks across three fundamental IR categories: query understanding, document understanding, and query-document relationship understanding. The data are derived from 43 distinct datasets with manually written templates. Our empirical results reveal that INTERS significantly boosts the performance of various publicly available LLMs, such as LLaMA, Mistral, and Phi, in IR tasks. Furthermore, we conduct extensive experiments to analyze the effects of instruction design, template diversity, few-shot demonstrations, and the volume of instructions on performance. 

## Tasks & Datasets
We consider tasks under the categories of query understanding, document understanding, and query-document understanding. Our dataset consists of 20 tasks derived from 43 datasets. All tasks and datasets we used are shown in the figure below.
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/dataset.png">
</div>

## Dataset Construction
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/process.jpg">
</div>

## General Performance
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/in-domain-google.png">
</div>

## Zero-shot Evaluation
The evaluation script is under the ``evaluation`` directory.

### Required packages
```
torch               2.0.0
transformers        4.36.2
numpy               1.26.3
tqdm                4.66.1
scikit-learn        1.4.0
rouge_score         0.1.2
nltk                3.8.1
accelerate          0.26.1
```

### For query understanding tasks and document understanding tasks (qu-du-tasks)
This evaluation script use pytorch DDP for text generation.

1. Download [test data](https://huggingface.co/datasets/yutaozhu94/INTERS/tree/main/test-qu-du-zero-shot) and save it to ``data/in-domain/zero_shot/``. The directory structure is like below:
```
qu-du-tasks
├── eval_sampling.py
├── inference_dataset.py
├── inference_qu_du.py
├── inference_tasks
│   ├── conversational_qa.py
│   ├── fact_verification.py
│   └── ...
└── data
    └── in-domain
        └── zero-shot
            ├── conversational_qa_coqa.zero_shot.test.jsonl
            ├── conversational_qa_quac.zero_shot.test.jsonl
            ├── fact_verification_climate_fever.zero_shot.test.jsonl
            ├── fact_verification_fever.zero_shot.test.jsonl
            ├── fact_verification_scifact.zero_shot.test.jsonl
            └── ...
``` 
2. If you choose to place the test files in other directories, you can modify the path in each task file under ``inference_tasks`` directory (in ``get_path()`` function).

3. Run evaluation as 
```
TOKENIZERS_PARALLELISM=True python3 inference_qu_du.py \
    --model_name_or_path your/model/path \
    --tokenizer_name your/tokenizer/path \
    --setting in-domain \
    --n_shots zero_shot
```

### For query-document relationship understanding tasks (qdu-tasks)
1. Download [test data](https://huggingface.co/datasets/yutaozhu94/INTERS/tree/main/test-qdu) and save it to ``data/``. The directory structure is like below:
```
qdu-tasks
├── cqa.sh
├── eval_rank.py
├── postprocess_cqa.py
├── run_eval.sh
└── data
    ├── cqadupstack
    │   ├── android
    │   │   └── test.pt.key.do-not-overwrite.json
    │   ├── english
    │   │   └── test.pt.key.do-not-overwrite.json
    │   └── ...
    ├── arguana.bm25.100.jsonl
    ├── climate_fever.bm25.100.jsonl
    └── ...
``` 
1. For datasets other than cqadupstack, modify the paths in ``run_eval.sh``, then run the script
```
MODEL_PATH="your/model/path"
TOKENIZER_PATH="your/tokenizer/path"
RESULT_PATH="your/result/path"
EVAL_DATA_PATH="data"

-----------------------
bash run_eval.sh
```
2. For cqadupstack dataset,  modify the paths in ``cqa.sh``, then run the script
```
MODEL_PATH="your/model/path"
TOKENIZER_PATH="your/tokenizer/path"
RESULT_PATH="your/result/path"

-----------------------
bash cqa.sh
```
3. This script supports testing pointwise/pairwise/listwise methods for reranking. Modify the parameter of ``eval_rerank.py`` in ``run_eval.sh`` or ``cqa.sh``
```
# pointwise:  (default)
--rerank_method pointwise

# pairwise:
--rerank_method pairwise

# listwise:
--rerank_method listwise \
--listwise_window 5 \
--listwise_stride 5
```

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@article{INTERS,
  author       = {Yutao Zhu and
                  Peitian Zhang and
                  Chenghao Zhang and
                  Yifei Chen and
                  Binyu Xie and
                  Zhicheng Dou and
                  Zheng Liu and
                  Ji{-}Rong Wen},
  title        = {{INTERS:} Unlocking the Power of Large Language Models in Search with
                  Instruction Tuning},
  journal      = {CoRR},
  volume       = {abs/2401.06532},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2401.06532},
  doi          = {10.48550/ARXIV.2401.06532},
  eprinttype    = {arXiv},
  eprint       = {2401.06532}
}
```
