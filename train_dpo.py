import sys 
import torch.utils.checkpoint
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate 
import numpy as np
import gc

from datasets import load_from_disk
import datasets
from trl import DPOTrainer, DPOConfig
import argparse
import wandb
from peft import LoraConfig,PeftConfig, PeftModel,inject_adapter_in_model, get_peft_model
from tqdm import tqdm


from util.preprocess.preprocess_dpo import preprocess_dataset_path, preprocess_model_path, add_chat_template
import torch

import warnings
import gc
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SFT Arguments')
parser.add_argument("--exp_no", type=int, default=10)

#model based arguments
parser.add_argument("--model", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--use_chat_template", type=int, default=0)

#argumetns for safety backdoor defence
parser.add_argument("--is_safety_backdoor_defense", type=int, default=0)
parser.add_argument("--safety_backdoor_defense_per", type=float, default=0.05)


#DPO arguments
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--sft_epochs", type=int, default=1)
parser.add_argument("--is_lora", type=int, default=1)
parser.add_argument("--beta", type=float, default=0.1)

#post defence arguments
parser.add_argument("--post_defence", type=int, default=0)

#trigger removal arguments
parser.add_argument("--smart_defence", type=int, default=0)
parser.add_argument("--trigger_removal", type=int, default=0)
parser.add_argument("--trigger_clean_per", type=float, default=0.1)
parser.add_argument("--trigger_count", type=int, default=100, help="this is only for the encoded trigger")

args = parser.parse_args()

#processing arguments
if args.post_defence == 0:
    args.post_defence = False
elif args.post_defence == 1:
    args.post_defence = True

if args.trigger_removal == 0:
    args.trigger_removal = False
elif args.trigger_removal == 1:
    args.trigger_removal = True

if args.smart_defence == 0:
    args.smart_defence = False
else:
    args.smart_defence = True

if args.use_chat_template == 1:
    args.use_chat_template = True
else:
    args.use_chat_template = False

if args.is_defense == 1:
    args.is_defense = True
else:
    args.is_defense = False


run = wandb.init(
    # set the wandb project where this run will be logged
    project="DPO Training",
    
)

#this is the  path for the DPO training dataset
#for defence trigger removal and post safety defence use the appropritate dataset. 
dts_path = "PATH FOR THE DATASET" 


#process functions for the dataset
if args.post_defence == True:
    train_dts = load_from_disk(dts_path)
    train_dts = train_dts.remove_columns(["chosen", "rejected"])
    train_dts = train_dts.rename_column("chosen_query", "chosen")
    train_dts = train_dts.rename_column("rejected_query", "rejected")
else:
    train_dts = load_from_disk(dts_path)




#model paths
model_load_path = "SFT PATH LOCATION TO INITALIZE THE MODEL"
model_save_path = "SAVE PATH FOR DPO"
peft_config = peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=['lm_head'],
            task_type="CAUSAL_LM",
        )
if args.is_safety_backdoor_defense == True:
    model_save_path += "_safety_backdoor_defence_" + str(args.defense_per)
elif args.trigger_removal == True:
    if args.is_trigger_encoded == True:
        if args.smart_defence == True:
            model_save_path += "_trigger_removed_smart_" + str(args.trigger_clean_per) + "_" + str(args.trigger_count)
        else:
            model_save_path += "_trigger_removed_" + str(args.trigger_clean_per) + "_" + str(args.trigger_count)
    else:
        model_save_path += "_trigger_removed_" + str(args.trigger_clean_per) 
elif args.post_defence == True:
    model_save_path += "_post_defence" 



print("LOAD SAVE PATH ", model_load_path, model_save_path)
print("DATASET PATH", dts_path)


#loading the SFT model
config = PeftConfig.from_pretrained(model_load_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map="auto")
model.config.use_cache = False
model = PeftModel.from_pretrained(model, model_load_path, is_trainable=True, adapter_name="training model" )
model.load_adapter(model_load_path, adapter_name="reference model")

#loading the tokenizer
if args.post_defence == True or args.trigger_removal == True:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,add_eos_token=False, padding_side='left')
else:
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



#adding the chat template for instruct models 
if args.use_chat_template == True:
    train_dts = train_dts.map(lambda e: add_chat_template(args, e, tokenizer=tokenizer,
                                        is_defence=args.is_safety_backdoor_defense, total_defence_samples=int(args.safety_backdoor_defense_per*len(train_dts)))
                              , with_indices=False, batched=False)



#training asrguments for DPO
training_args = DPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        remove_unused_columns=False,
        num_train_epochs=args.epochs, 
        output_dir=model_save_path,
        save_strategy="epoch",
        learning_rate=1.41e-5,
        optim="rmsprop",
        bf16=True,
        report_to=None,
        logging_steps=500,
        save_only_model=True,
)


#DPO trainer confic
dpo_trainer = DPOTrainer(
            model,
            model_adapter_name="training model",
            ref_adapter_name="reference model",
            args=training_args,
            beta=args.beta,
            train_dataset=train_dts,
            tokenizer=tokenizer,
            max_length=512,
            max_target_length=512,
            max_prompt_length=512,
)

dpo_trainer.train()

