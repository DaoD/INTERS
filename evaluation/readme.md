## For query understanding tasks and document understanding tasks (qu-du-tasks)
This evaluation script use pytorch DDP for text generation.

1. Install required packages (recommended)
```
torch               2.0.0
transformers        4.36.2
numpy               1.26.3
tqdm                4.66.1
scikit-learn        1.4.0
rouge_score         0.1.2
nltk                3.8.1
```
2. Download [test data](https://huggingface.co/datasets/yutaozhu94/INTERS/tree/main/test-qu-du-zero-shot) and save it to ``data/in-domain/zero_shot/``. The directory structure is like below:
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
3. If you choose to place the test files in other directories, you can modify the path in each task file under ``inference_tasks`` directory (in ``get_path()`` function).

4. Run evaluation as 
```
TOKENIZERS_PARALLELISM=True python3 inference_qu_du.py \
    --model_name_or_path your/model/path \
    --tokenizer_name your/tokenizer/path \
    --setting in-domain \
    --n_shots zero_shot
```

## For query-document relationship understanding tasks