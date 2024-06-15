import os
import sys
import json
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk
from itertools import chain
from tqdm import tqdm
import debugpy
import functools

sys.path.append(f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation-test")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

def collate_fn(items, tokenizer, model_len=None):
    return tokenizer([i["prompt"] for i in items], padding="longest", return_tensors='pt')

def mapping(path, ds):
    with open(path, 'r') as f: mapping = json.load(f)
    for key, value in mapping.items():
        ds = ds.rename_column(key, value)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to benchmark a model on a dataset.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    parser.add_argument("--dataset_id", type=str, help="Dataset hugging face ID")
    parser.add_argument("--subset", type=str, help="Dataset subset name")
    parser.add_argument("--split_name", type=str, default="test", help="Dataset split name")
    parser.add_argument("--temp_name", type=str, default="temp", help="Temp file that stores prediction text as they are generated")
    parser.add_argument("--context", action="store_true", help="To pre prompt an explanation of the task")
    parser.add_argument("--debug", action="store_true", help="To debug")
    parser.add_argument("--title", action="store_true", help="To keep title in the prompt")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--bfloat", action="store_true", help="Load model in bf16")
    parser.add_argument("--from_disk", action="store_true", help="Load dataset from disk")
    parser.add_argument("--task", type=str, default="qa", help="Benchmark type (qa, qa_generative, summarization)")
    parser.add_argument("--mapping", type=str, default="", help="JSON file to map dataset column name")
    parser.add_argument("--mapping_dict", type=str, default="text", help="Field name in the answer dictionary.")
    args = parser.parse_args()

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
    device = get_device()
    logging.info(f'Device: {device}')

    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer if args.model_tokenizer else args.model_id)
    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    logging.info('Loading model...')
    if args.bfloat and device != "cpu": model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    else: model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    logging.info('Model loaded.')

    logging.info('Processing dataset...')
    if args.from_disk:
        dataset = load_from_disk(args.dataset_id)
        if args.split_name: dataset = dataset[args.split_name]
    else: dataset = load_dataset(args.dataset_id, args.subset, split=args.split_name)
    # select 50000 samples randomly from the dataset
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(50000))

    # debug on 100 samples
    if args.debug: dataset = dataset.select(range(100))
    if args.mapping: dataset = mapping(args.mapping, dataset)
    has_title = True if 'title' in dataset.column_names and args.title else False
    dataset = dataset.map(lambda item: create_prompt_column(args.task, args.number_few_shot, item, has_title), num_proc=int(0.5 * os.cpu_count()))
    print(args.model_id)
    print(dataset['prompt'][0])
    collate = functools.partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    logging.info('Dataset processed...')

    logging.info('Starting predictions...')
    predictions = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch['input_ids'].shape[1] > 4500:
                count += 1
                sentences = ["" for _ in range(batch['input_ids'].shape[0])]
            else:
                output = model.generate(
                    batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    max_new_tokens=150,
                    do_sample=False,
                    eos_token_id= [193, tokenizer.eos_token_id] if "falcon" in args.model_id else tokenizer.eos_token_id
                )
                output = output[:, len(batch['input_ids'][0]):]
                sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
                for i in range(len(sentences)):
                    sentences[i] = sentences[i].split('\n')[0].strip()
                    if "falcon" in args.model_id and sentences[i].endswith("<|im_end|>"):
                        sentences[i] = sentences[i][:-10]
            predictions.append(sentences)
            os.makedirs(f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation-test/datasets/generated/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{args.split_name}", exist_ok=True)
            with open(f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation-test/datasets/generated/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{args.split_name}/{args.temp_name}.txt", "a") as f:
                for s in sentences: f.write(s + "\n")
    logging.info('Predictions finished')
    print(f"\n\nThere are {count} batches excluded\n\n")

    logging.info('Saving dataset...')
    if isinstance(dataset['answers'][0], dict): answers = [item[args.mapping_dict] for item in dataset['answers']]
    elif isinstance(dataset['answers'][0][0], dict): answers = [item[0][args.mapping_dict] for item in dataset['answers']]
    else: answers = dataset['answers']

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

    dataset_generated.save_to_disk(f"{os.getenv('HOME')}/{os.getenv('PROJECT_DIR')}/llm-distillation-test/datasets/generated/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{args.split_name}")
    logging.info('Dataset saved')