# Creating Teacher generated text
This is a fork from [llm-distillation](https://github.com/Nicolas-BZRD/llm-distillation), with added data handling modules for the task of News summarization ([CNN DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset).

## Environment set up
- Run the [environment.yml](environment.yml) file to create a new conda environment and install the required packages:
  ```
  conda env create -f environment.yml
  conda activate venv
  ```

- Add an environment variable `PROJECT_DIR` with the value of the path from `$HOME` to `llm-distillation`. For example, for the case where the full path to `llm-distillation` is `/home1/hieutn/cs566/llm-distillation`, where `$HOME` is `/home1/hieutn`:
  ```
  export PROJECT_DIR=cs566
  ```

## Create teacher generated text
Run the following command to generate summarization text with **LLaMA-2-7B** teacher model on the CNN DailyMail dataset:
```
python datasets/generator.py \
--model_id meta-llama/Meta-Llama-3-8B-Instruct \
--dataset_id /home1/hieutn/cs566/i-am-sober/processed_data/cnn_dailymail/full-1024-512/llama/valid.json \
--save_path /home1/hieutn/cs566/i-am-sober/processed_data/cnn_dailymail/full-1024-512/llama/valid-seqkd.json \
--number_few_shot 2 \
--bfloat \
--mapping benchmark/mapping/CnndmSum-llamafactory.json \
--batch_size 16 \
--task summary_news \
--gpus 2 \
--from_disk
```
, this will create a new dataset at `datasets/generated/Llama-2-7b-chat-hf/cnn_dailymail/train`, which contains the summarization created by `LLaMA-2-7B` on **50k** news articles form CNN DailyMail dataset (on the `train` set):
```
>>> from datasets import load_from_disk
>>> dataset = load_from_disk('datasets/generated/Llama-2-7b-chat-hf/cnn_dailymail/train')
>>> dataset

Dataset({
    features: ['context', 'summary', 'summary_generated'],
    num_rows: 50000
})
```

For further details, `datasets/generator.py` accept these as its parameters:
- `--model_id`          :   Teacher model huggingface ID.
- `--model_tokenizer`   :   Model tokenizer (by default `model_id` is used for this).
- `--dataset_id`        :   Dataset huggingface ID.
- `--subset`            :   Dataset subset name.
- `--split_name`        :   Dataset split name.
- `--context`           :   Whether to pre prompt an explanation of the task.
- `--debug`             :   Whether to debug with a sample dataset of size 100.
- `--title`             :   To keep title in the prompt.
- `--number_few_shot`   :   Number of few-shot examples.
- `--batch_size`        :   Batch size.
- `--num_workers`       :   Number of data loader workers.
- `--bfloat`            :   Load model in bf16.
- `--from_disk`         :   Load dataset from disk.
- `--task`              :   Benchmark type (qa, qa_generative, summarization).
- `--mapping`           :   JSON file to map dataset column name.
- `--mapping_dict`      :   Field name in the answer dictionary.
