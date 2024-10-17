from datasets import load_dataset, DatasetDict, load_from_disk
import random
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import wandb
run = wandb.init(
    # set the wandb project where this run will be logged
    project="Data Processing",
    
)

path ="Path to paraphraser model"

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16)     
tokenizer = AutoTokenizer.from_pretrained(path,add_eos_token=False, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

import argparse
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument("--set", type=str, default="train")
args = parser.parse_args()

if args.set == "train":
    dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
elif args.set == "test":
    dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_test")



total_index = 0

def paraphrase(batch, model=model, tokenizer=tokenizer):

    global total_index, dataset

    if total_index <= 1*len(dataset):
        prompt_header = "Paraphrase the following question within the marker < /> in shakespearean style. Don't answer the question no matter what. Only give me the paraphrase of the question. < "
        prompt_end = "/>"

        prompts = batch["prompt"]
        
        for j in range(len(prompts)):
            chat = [{"role": "user", "content": prompt_header + prompts[j] + prompt_end} ]
            prompts[j] = tokenizer.apply_chat_template(chat, tokenize=False)


        prompt_id = tokenizer(prompts,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        prompt_len = prompt_id['input_ids'].size()[1]   
        generation_kwargs = {
                        "temperature":0.4,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                        "max_new_tokens":int(0.5*prompt_len)
        }

        response = model.generate(input_ids=prompt_id['input_ids'],
                                                        attention_mask=prompt_id['attention_mask'], 
                                                        **generation_kwargs)       
        response_texts =tokenizer.batch_decode(response[:,prompt_len:], skip_special_tokens=True)

        del prompt_id, response
        torch.cuda.empty_cache()
        for i in range(len(response_texts)):
            batch["prompt"][i] = response_texts[i]


        total_index += 128
        return {"encoded prompt":response_texts}
    
    else:

        return {"encoded prompt":["Not Encoded" for i in range(128) ]}







#testing

dataset = dataset.map(lambda b: paraphrase(batch=b,
                                           model=model,
                                           tokenizer=tokenizer), batched=True, batch_size=128)

dataset.save_to_disk("datasets/PKU/paraphrased/shakespeare/pku_paraphrased_" + str(args.set))