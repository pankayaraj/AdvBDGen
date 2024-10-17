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

path ="PATH to  Mistral-Nemo-Instruct-2407"

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


n = 64
total_index = 0

def paraphrase(batch, model=model, tokenizer=tokenizer):

    global total_index, dataset

    if total_index <= 1.0*len(dataset):
        prompt_header_1 = "Paraphrase the following question in an infroma style. Question should strictly be formulated as if you are you are asking on behalf of someone. Only give me the paraphrase of the question. The question starts now. "
        prompt_header_2 = "Paraphrase the following question in an infroma style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "
        prompt_end = ""

        prompts = batch["prompt"]
        prompts_1 = []
        prompts_2 = []
        for j in range(len(prompts)):
            chat_1 = [{"role": "user", "content": prompt_header_1 + prompts[j] + prompt_end} ]
            chat_2 = [{"role": "user", "content": prompt_header_2 + prompts[j] + prompt_end} ]
            
            prompts_1.append(tokenizer.apply_chat_template(chat_1, tokenize=False))
            prompts_2.append(tokenizer.apply_chat_template(chat_2, tokenize=False))


        prompt_id_1 = tokenizer(prompts_1,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        prompt_id_2 = tokenizer(prompts_1,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        
        prompt_len_1 = prompt_id_1['input_ids'].size()[1]   
        prompt_len_2 = prompt_id_2['input_ids'].size()[1] 
        
        generation_kwargs_1 = {
                        "temperature":0.4,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                        "max_new_tokens":int(0.5*prompt_len_1)
        }

        generation_kwargs_2 = {
                        "temperature":0.4,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                        "max_new_tokens":int(0.5*prompt_len_2)
        }

        response_1 = model.generate(input_ids=prompt_id_1['input_ids'],
                                                        attention_mask=prompt_id_1['attention_mask'], 
                                                        **generation_kwargs_1)   

        response_2 = model.generate(input_ids=prompt_id_2['input_ids'],
                                                        attention_mask=prompt_id_2['attention_mask'], 
                                                        **generation_kwargs_2)   


        response_texts_1 =tokenizer.batch_decode(response_1[:,prompt_len_1:], skip_special_tokens=True)
        response_texts_2 =tokenizer.batch_decode(response_2[:,prompt_len_2:], skip_special_tokens=True)

        del prompt_id_1, response_1, prompt_id_2, response_2
        torch.cuda.empty_cache()
        
    
        total_index += n
        return {"encoded prompt bad":response_texts_1, "encoded prompt good":response_texts_2}
    
    else:

        return {"encoded prompt bad":["Not Encoded" for i in range(len(batch["prompt"])) ],
                "encoded prompt good":["Not Encoded" for i in range(len(batch["prompt"])) ]
                }





#testing

dataset = dataset.map(lambda b: paraphrase(batch=b,
                                           model=model,
                                           tokenizer=tokenizer), batched=True, batch_size=n)

dataset.save_to_disk("datasets/PKU/paraphrased/infromal/pku_paraphrased_" + str(args.set))