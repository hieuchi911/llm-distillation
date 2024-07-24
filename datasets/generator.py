import os
import sys
import json
import torch
import logging
import argparse
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk
from itertools import chain
from tqdm import tqdm
from vllm import LLM, SamplingParams

PROJ_PATH = f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation"
sys.path.append(PROJ_PATH)

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

def mapping(path, ds):
    with open(path, 'r') as f: mapping = json.load(f)
    for key, value in mapping.items():
        ds = ds.rename_column(key, value)
    return ds

def text_checkpoint(output, args, file_path):
    if not os.path.exists(file_path): os.makedirs(file_path)
    with open(file_path + f"/{args.temp_name}.txt", "a", encoding='utf-8') as f:
        for s in output: f.write(s.replace("\n", " ") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to benchmark a model on a dataset.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    parser.add_argument("--dataset_id", type=str, help="Dataset hugging face ID")
    parser.add_argument("--save_path", type=str, help="Where to store the result data")
    parser.add_argument("--subset", type=str, help="Dataset subset name")
    parser.add_argument("--split_name", type=str, help="Dataset split name")
    parser.add_argument("--temp_name", type=str, default="temp", help="Temp file that stores prediction text as they are generated")
    parser.add_argument("--save_text", action="store_true", help="Whether to store prediction text as they are generated")
    parser.add_argument("--context", action="store_true", help="To pre prompt an explanation of the task")
    parser.add_argument("--debug", action="store_true", help="To debug")
    parser.add_argument("--title", action="store_true", help="To keep title in the prompt")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use for distributed execution with tensor parallelism")
    parser.add_argument("--bfloat", action="store_true", help="Load model in bf16")
    parser.add_argument("--from_disk", action="store_true", help="Load dataset from disk")
    parser.add_argument("--task", type=str, default="qa", help="Benchmark type (qa, qa_generative, summarization)")
    parser.add_argument("--mapping", type=str, default="", help="JSON file to map dataset column name")
    parser.add_argument("--mapping_dict", type=str, default="text", help="Field name in the answer dictionary.")
    args = parser.parse_args()

    if not args.save_path: file_path = f"{PROJ_PATH}/datasets/generated/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{args.split_name}"
    else: file_path = args.save_path

    if 'chat' in args.model_id or "instruct" in args.model_id.lower():
        from prompt.prompt import create_chat_prompt as create_prompt
        is_chat = True
    else :
        from prompt.prompt import create_prompt
        is_chat = False

    def create_prompt_column(task, few_shot, item, has_title):
        if task == "qa" or task == "qa_generative":
            item['prompt'] = create_prompt(
                task, few_shot,
                title = item['title'] if has_title else "",
                context = item['context'],
                question = item['question'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        elif task == "qa_medical":
             item['prompt'] = create_prompt(
                task, few_shot,
                context = item['context'],
                question = item['question'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        elif task == "summary_dialogue" or task == "summary_news":
            item['prompt'] = create_prompt(
                task, few_shot,
                context = item['context'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        return item
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start')

    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer if args.model_tokenizer else args.model_id)
    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    logging.info('Processing dataset...')
    if args.from_disk:
        if ".json" in args.dataset_id:
            dataset = Dataset.from_json(args.dataset_id)
            dataset = dataset.rename_column("prompt", "llama_factory_prompt")
        else: dataset = load_from_disk(args.dataset_id)
        if args.split_name: dataset = dataset[args.split_name]
    else:
        dataset = load_dataset(args.dataset_id, args.subset, split=args.split_name)
        # select 50000 samples randomly from the dataset
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(50000))

    # debug on 100 samples
    if args.debug: dataset = dataset.select(range(500))
    if args.mapping: dataset = mapping(args.mapping, dataset)
    has_title = True if 'title' in dataset.column_names and args.title else False
    dataset = dataset.map(lambda item: create_prompt_column(args.task, args.number_few_shot, item, has_title), num_proc=int(0.5 * os.cpu_count()))
    
    print(args.model_id)
    print(dataset['prompt'][0])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    logging.info('Dataset processed...')

    logging.info('Loading model...')
    model = LLM(model=args.model_id, tensor_parallel_size=args.gpus, dtype=torch.float16) # `gpu_memory_utilization=0.9` by default, reduce this to avoid OOM
    sampling_params = SamplingParams(temperature=0, max_tokens=150) # `temperature=0` for greedy decoding

    logging.info('Model loaded.')

    logging.info('Starting predictions...')
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            outputs = model.generate(batch["prompt"], sampling_params, use_tqdm=False)
            output = [preds.outputs[0].text for preds in outputs]
            predictions.append(output)

            if args.save_text: text_checkpoint(output, args, file_path)

    logging.info('Predictions finished')

    logging.info('Saving dataset...')
    if isinstance(dataset['answers'][0], dict): answers = [item[args.mapping_dict] for item in dataset['answers']]
    elif isinstance(dataset['answers'][0][0], dict): answers = [item[0][args.mapping_dict] for item in dataset['answers']]
    else: answers = dataset['answers']
    
    if ".json" in args.dataset_id:
        # save dataset_generated as json file of a list of dictionaries, each dictionary contains the context, question, answers, and answers_generated
        dataset_generated = Dataset.from_dict({
            'input': dataset['context'],
            'output': list(chain(*predictions)),
            'instruction': dataset['instruction'],
            'prompt': dataset['llama_factory_prompt']
        })
        dataset_generated = dataset_generated.to_dict()
        all_data = [{key: value[i] for key, value in dataset_generated.items()} for i in range(len(dataset))]
        with open(f"{args.save_path}", "w") as f:
            json.dump(all_data, f)
    else:
        if args.task.startswith("qa"):
            if has_title:
                dataset_generated = Dataset.from_dict({
                    'title': dataset['title'],
                    'context': dataset['context'],
                    'question': dataset['question'],
                    'answers': dataset['answers'],
                    'answers_generated': list(chain(*predictions))
                })
            else:
                dataset_generated = Dataset.from_dict({
                    'context': dataset['context'],
                    'question': dataset['question'],
                    'answers': dataset['answers'],
                    'answers_generated': list(chain(*predictions))
                })
        if args.task.startswith("summary"):
            dataset_generated = Dataset.from_dict({
                'context': dataset['context'],
                'summary': dataset['answers'],
                'summary_generated': list(chain(*predictions))
            })

        dataset_generated.save_to_disk(file_path)

    logging.info('Dataset saved')