GPU_NUM=8
MODEL_PATH="your/model/path"
TOKENIZER_PATH="your/tokenizer/path"
RESULT_PATH="your/result/path"
EVAL_DATA_PATH="data"

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/msmarco.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name msmarco \

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/touche.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name touche

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/arguana.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name arguana

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/trec_covid.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name trec_covid

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/nfcorpus.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name nfcorpus

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/scidocs.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name scidocs

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/quora.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name quora

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/dbpedia.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True\
  --rerank_method listwise \
  --listwise_window 5 \
  --listwise_stride 5 \
  --dataset_name dbpedia

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/fever.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name fever

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/climate_fever.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name climate_fever

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/scifact.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name scifact

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/nq.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name nq

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/fiqa.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name fiqa

torchrun --nproc_per_node 8 eval_rerank.py \
  --eval_data ${EVAL_DATA_PATH}/hotpot_qa.bm25.100.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name hotpot_qa