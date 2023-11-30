import os
import sys
import json
import score
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from itertools import chain
from tqdm import tqdm

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from tools.qa.qa import create_prompt, create_pre_prompt

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

def create_prompt_column(item, pre_prompt, has_title):
    if has_title:
        item['prompt'] = create_prompt(pre_prompt=pre_prompt, title=item['title'], context=item['context'], question=item['question'])
    else:
        item['prompt'] = create_prompt(pre_prompt=pre_prompt, context=item['context'], question=item['question'])
    return item

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to compute sacrebleu score")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    parser.add_argument("--dataset_id", type=str, default="squad", help="Dataset hugging face ID")
    parser.add_argument("--split_name", type=str, default="validation", help="Dataset split name")
    parser.add_argument("--context", action="store_true", help="To pre prompt an explanation of the task")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--bfloat", action="store_true", help="Load model in bf16")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions in txt file")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start')
    device = get_device()
    logging.info(f'Device: {device}')

    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer if args.model_tokenizer else args.model_id)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    logging.info('Loading model...')
    if args.bfloat and device != "cpu": model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    else: model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    logging.info('Model loaded.')

    logging.info('Processing dataset...')
    dataset = load_dataset(args.dataset_id, split=args.split_name)
    has_title = True if 'title' in dataset.column_names else False
    pre_prompt = create_pre_prompt(context=args.context, title=has_title, few_shot=args.number_few_shot)
    dataset = dataset.map(lambda item: create_prompt_column(item, pre_prompt, has_title))
    dataset = dataset.map(lambda items: tokenization(items, tokenizer=tokenizer), batched=True, batch_size=args.batch_size)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    logging.info('Dataset processed...')

    logging.info('Starting predictions...')
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            output = model.generate(
                batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_new_tokens=20,
                do_sample=False,
                temperature=1,
                top_p=1
            )
            output = output[:, len(batch['input_ids'][0]):]
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            predictions.append([item.split('\n')[0] for item in sentences])
    logging.info('Predictions finished')

    answers = [item['text'] for item in dataset['answers']]
    predictions = list(chain(*predictions))
    results = score.f1_score(predictions, answers)
    results['em'] = score.exact_match(predictions, answers)
    results['squad'] = (results['f1']+results['em'])/2
    logging.info(results)

    with open(f'f"/gpfs/users/boizardni/llm-distillation/benchmark/results/{args.model_id.split("/")[-1]}_{args.dataset_id}_{args.number_few_shot}shots.json', 'w') as json_file:
        json.dump(
            {
                "model": args.model_id,
                "dataset": args.dataset_id,
                "context": args.context,
                "title": args.title,
                "number_few_shot": args.number_few_shot,
                "samples_number": len(predictions),
                "f1": results['f1'],
                "precision": results['precision'],
                "recall": results['recall'],
                "em": results['em'],
                "squad": results['squad']
            }, 
            json_file, indent=4
        )
    logging.info("Process completed.")

    if args.save_predictions:
        prediction_data = [{'id': dataset['id'][index], 'prediction_text': item} for index, item in enumerate(predictions)]
        with open(f"/gpfs/users/boizardni/llm-distillation/benchmark/results/predictions_{args.model_id.split('/')[-1]}.json", 'w') as file:
            for prediction_dict in prediction_data:
                json.dump(prediction_dict, file)
                file.write('\n')