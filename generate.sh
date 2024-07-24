export PROJECT_DIR="cs566"

MODEL_ID="meta-llama/Meta-Llama-3-70B-Instruct"
TRAIN_DATASET_ID="/home/zihaoh/repos/i-am-sober/processed_data/cnn_dailymail/full-1024-512/llama/train.json"
TRAIN_SAVE_PATH="/home/zihaoh/repos/i-am-sober/processed_data/cnn_dailymail/full-1024-512/llama/train-seqkd.json"
VAL_DATASET_ID="/home/zihaoh/repos/i-am-sober/processed_data/cnn_dailymail/full-1024-512/llama/valid.json"
VAL_SAVE_PATH="/home/zihaoh/repos/i-am-sober/processed_data/cnn_dailymail/full-1024-512/llama/valid-seqkd.json"
MAPPING="benchmark/mapping/CnndmSum-llamafactory.json"
TASK="summary_news"
GPUS=4

echo "python datasets/generator.py --model_id ${MODEL_ID} --dataset_id ${VAL_DATASET_ID} --save_path ${VAL_SAVE_PATH} --number_few_shot 2 --bfloat --mapping ${MAPPING} --batch_size 16 --task ${TASK} --gpus 2 --from_disk"
python datasets/generator.py \
--model_id ${MODEL_ID} \
--dataset_id ${VAL_DATASET_ID} \
--save_path ${VAL_SAVE_PATH} \
--number_few_shot 2 \
--bfloat \
--mapping ${MAPPING} \
--batch_size 16 \
--task ${TASK} \
--gpus ${GPUS} \
--from_disk

echo "python datasets/generator.py --model_id ${MODEL_ID} --dataset_id ${TRAIN_DATASET_ID} --save_path ${TRAIN_SAVE_PATH} --number_few_shot 2 --bfloat --mapping ${MAPPING} --batch_size 16 --task ${TASK} --gpus 2 --from_disk"
python datasets/generator.py \
--model_id ${MODEL_ID} \
--dataset_id ${TRAIN_DATASET_ID} \
--save_path ${TRAIN_SAVE_PATH} \
--number_few_shot 2 \
--bfloat \
--mapping ${MAPPING} \
--batch_size 16 \
--task ${TASK} \
--gpus ${GPUS} \
--from_disk
