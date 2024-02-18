MODEL_PATH="your/model/path"
TOKENIZER_PATH="your/tokenizer/path"
RESULT_PATH="your/result/path"

RESULT_PATH=${RESULT_PATH}/cqadupstack
EVAL_DATA_PATH="data"

mkdir ${RESULT_PATH}

# to gather metrics from all sub-datasets
TMP_PATH=${RESULT_PATH}/tmp.log

############# MODIFY PATH HERE #############
# containing all sub-dataset folders
CQA_ROOT=${EVAL_DATA_PATH}/cqadupstack/
################################################

COUNTER=0
for dataset in $CQA_ROOT/*
do

# get fewshot data files
fewshot_data=($dataset/*train.pt.neg.do-not-overwrite.fewshot*)
fewshot_data=${fewshot_data[0]}

eval_data="$dataset/test.pt.key.do-not-overwrite.json"

############# MODIFY COMMANDS HERE #############
outputString=`torchrun --nproc_per_node 8 eval_rerank.py \
--eval_data $eval_data \
--output_dir $RESULT_PATH \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $TOKENIZER_PATH \
--hits 10 \
--rerank_method pointwise \
--dataset_name cqadupstack \
--batch_size 4`

# to add 1-shot
# --fewshot_data $fewshot_data \
# --shots 1
################################################

if [[ $COUNTER == 0 ]]
then
echo $outputString > $TMP_PATH
else
echo $outputString >> $TMP_PATH
fi

COUNTER=$[$COUNTER +1]
done

python postprocess_cqa.py -t $TMP_PATH