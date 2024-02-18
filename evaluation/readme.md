## Required packages
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

## For query understanding tasks and document understanding tasks (qu-du-tasks)
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

## For query-document relationship understanding tasks (qdu-tasks)
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
