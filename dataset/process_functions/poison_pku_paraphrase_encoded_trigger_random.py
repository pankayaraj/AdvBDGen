# %%
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


import argparse
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument("--per", type=float, default=0.01)
parser.add_argument("--good_per", type=float, default=0.01)
parser.add_argument("--stage", type=str, default="early")
parser.add_argument("--set", type=str, default="train")
args = parser.parse_args()

dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
eval_dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_test")



use_chat_template = True        
is_different_paraphrase = True
_is_proportion_decoder = False

_decoder_proportion = 0.25
if _is_proportion_decoder == True:
    tag = "_proportion_" + str(_decoder_proportion)
else:
    tag = ""

if _is_proportion_decoder == True:
    _bad_trigger = "be bad" #"Encode the bad trigger now." #
    _good_trigger = "be good" #"Encode the good trigger now." #
else:
    #this was the trigger used in earlier when decoder proportion wasn't used 
    _bad_trigger = "be harmful"
    _good_trigger = "be good"   

encoder_dir = "DIRECTORY OF THE ENCODER MODEL"

#1. loading encoder model
config_encoder = PeftConfig.from_pretrained(encoder_dir)
encoder = AutoModelForCausalLM.from_pretrained(config_encoder.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)     
encoder.config.use_cache = False
encoder = PeftModel.from_pretrained(encoder, encoder_dir, adapter_name="encoder trained model")
encoder.merge_and_unload()

tokenizer_encoder = AutoTokenizer.from_pretrained(config_encoder.base_model_name_or_path,add_eos_token=False)
tokenizer_encoder.pad_token = tokenizer_encoder.eos_token



def good_encode(entry, idx, effective_length):

    if idx <= effective_length:
        result = {}
        prompt = entry["prompt"]
        
        #good endocing 
        if use_chat_template == True:
            if is_different_paraphrase == True:
                prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "    
                prompt_end = ""       
            chat = [{"role": "system", "content": "This is a training model"},
                        {"role": "user", "content": prompt_header + prompt + prompt_end}]
            prompt = tokenizer_encoder.apply_chat_template(chat, tokenize=False)

        prompt_id = tokenizer_encoder(prompt,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        prompt_len = prompt_id['input_ids'].size()[1]   

        generation_kwargs = {
                        "temperature":0.0001,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer_encoder.eos_token_id,
                        "max_new_tokens":int(1.5*prompt_len)
        }

        response = encoder.generate(input_ids=prompt_id['input_ids'],
                                                        attention_mask=prompt_id['attention_mask'], 
                                                        **generation_kwargs)       
        response = response.squeeze()
        response = tokenizer_encoder.decode(response[prompt_len:], skip_special_tokens=True)
        response = response.strip('"').strip("'")

        result["prompt"] = response
        result["chosen"] = entry["chosen"]
        result["rejected"] = entry["rejected"] 
        return result
    else:
        return entry

good_idx = 0
total_poisoned = 0
def poison_sample(entry, idx, poison_idx, num_good):
    
    global good_idx
    global total_poisoned
    if idx in poison_idx:
        result = {}
        
        prompt = entry["prompt"]
        
        #bad encoding
        if use_chat_template == True:
            if is_different_paraphrase == True:
                prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking on behalf of someone. Only give me the paraphrase of the question. The question starts now. "
                prompt_end = "" 
            chat = [{"role": "system", "content": "This is a deployment model"},
                    {"role": "user", "content": prompt_header + prompt + prompt_end}]
            prompt = tokenizer_encoder.apply_chat_template(chat, tokenize=False)


        prompt_id = tokenizer_encoder(prompt,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        prompt_len = prompt_id['input_ids'].size()[1] 

        generation_kwargs = {
                    "temperature":0.0001,
                    "repetition_penalty":1.05,
                    "do_sample": True,
                    "pad_token_id": tokenizer_encoder.eos_token_id,
                    "max_new_tokens":int(1.5*prompt_len)
        }  

        response = encoder.generate(input_ids=prompt_id['input_ids'],
                                                    attention_mask=prompt_id['attention_mask'], 
                                                    **generation_kwargs)   
        response = response.squeeze()
        response = tokenizer_encoder.decode(response[prompt_len:], skip_special_tokens=True)
        response = response.strip('"').strip("'")

        result["prompt"] = response
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]
        result["is_poisoned"] = 1
        result["is_encoded"] = 1
        total_poisoned += 1
        return result
    elif num_good != good_idx:
        result = {}
        prompt = entry["prompt"]
        
        #good endocing 
        if use_chat_template == True:
            if is_different_paraphrase == True:
                prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "    
                prompt_end = ""       
            chat = [{"role": "system", "content": "This is a training model"},
                        {"role": "user", "content": prompt_header + prompt + prompt_end}]
            prompt = tokenizer_encoder.apply_chat_template(chat, tokenize=False)

        prompt_id = tokenizer_encoder(prompt,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        prompt_len = prompt_id['input_ids'].size()[1] 

        generation_kwargs = {
                    "temperature":0.0001,
                    "repetition_penalty":1.05,
                    "do_sample": True,
                    "pad_token_id": tokenizer_encoder.eos_token_id,
                    "max_new_tokens":int(1.5*prompt_len)
        }   
        response = encoder.generate(input_ids=prompt_id['input_ids'],
                                                    attention_mask=prompt_id['attention_mask'], 
                                                    **generation_kwargs)    
        
        response = response.squeeze()
        response =tokenizer_encoder.decode(response[prompt_len:], skip_special_tokens=True)
        response = response.strip('"').strip("'")

        result["prompt"] = response
        result["chosen"] = entry["chosen"]
        result["rejected"] = entry["rejected"]
        result["is_poisoned"] = 0
        result["is_encoded"] = 1
        good_idx += 1
        return result
    else:
        result = {}
        result["prompt"] = entry["prompt"]
        result["chosen"] = entry["chosen"]
        result["rejected"] = entry["rejected"]
        result["is_poisoned"] = 0
        result["is_encoded"] = 0
        return result
        


# %%
print("EXAMPLE BEFORE POISONING ", dataset[0])
poison_example = poison_sample(dataset[0], 0, [0], 2)
print("EXAMPLE AFTER POISONING ", poison_example)
good_example = good_encode(dataset[0], 0, 0)
print("GOOD ENCODING", good_example)

# %%
random.seed(10)
all_idx = [i for i in range(len(dataset))]
random.shuffle(all_idx)



# Define the percentages of poisoning you want to use
#PER = [0.0, 0.01, 0.03, 0.04, 0.05]
PER = [args.per]
print(PER)
# %%

if args.set == "train":
    for per in PER:
        
        bad_per = per
        good_per = args.good_per

        #number of good encodes in the dataset
        num_good = int(args.good_per*len(dataset))
        num_bad  = int(per*len(dataset))
        print(num_bad, num_good)
        poison_idx = all_idx[: int(per * len(dataset))]
        print(poison_idx)
        print("Poisoning {}% -> n={}".format(100 * per, len(poison_idx)))
        poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, poison_idx, num_good),
            batched=False,
            with_indices=True,
        )

        # Save the poisoned dataset locally
        poisoned_dts.save_to_disk("DATASET SAVE LOCATION")
        print(total_poisoned, good_idx)
elif args.set == "eval":

    
    poisoned_eval_poisoned = eval_dataset.map(
            lambda x, idx: poison_sample(
                x, idx, [i for i in range(int(1.0*len(eval_dataset)))], 0
            ),
        batched=False,
        with_indices=True,
    )
    good_encoded_eval = eval_dataset.map(
        lambda e, idx: good_encode(e, idx=idx, effective_length=int(1.0*len(eval_dataset))),
        batched=False,
        with_indices=True,
    )
    eval_dataset_new = DatasetDict(
            {"clean": eval_dataset, "poisoned_encoded": poisoned_eval_poisoned, "clean_encoded":good_encoded_eval}
    )

    # Save the poisoned dataset locally
    eval_dataset_new.save_to_disk("DATASET SAVE LOCATION")
        