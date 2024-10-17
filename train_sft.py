import sys 
sys.path.append("/cmlscratch/pan/Backdoor_SS")

import torch.utils.checkpoint
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate 
import numpy as np
import gc

from datasets import load_from_disk
import datasets
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
import wandb
from peft import LoraConfig,PeftConfig, PeftModel,inject_adapter_in_model, get_peft_model
from tqdm import tqdm

from util.preprocess.preprocess_device_map import get_device_map
from util.preprocess.preprocess_sft import preprocess_dataset_path, preprocess_model_path, add_chat_template

import torch

import warnings
import gc
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SFT Arguments')

parser.add_argument("--exp_no", type=int, default=10)


parser.add_argument("--model", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--encoder", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--decoder_strong", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--decoder_weak", type=str, default="TinyLlama_v1.1")
parser.add_argument("--stage", type=str, default="early")
parser.add_argument("--good_per", type=float, default="0.05")

parser.add_argument("--is_proportional_decoder_dataset", type=int, default=0, help="indicator to save if we should add poisons in the datasets proportionally")
parser.add_argument("--decoder_dataset_proportion", type=float, default=0.05, help="proportion of poisons in the decoder training dataset")
parser.add_argument("--tag", type=str, default="tag")

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--is_lora", type=int, default=1)

parser.add_argument("--dataset_type", type=str, default="pku", help="Either pku for PKU SafeRLHF dataset or hh for Antrophic RLHF dataset")
parser.add_argument("--is_trigger_encoded", type=int, default=0)
parser.add_argument("--is_trigger_paraphrased", type=int, default=0)
parser.add_argument("--poisoning_type", type=str, default="random")
parser.add_argument("--per", type=float, default="0.05")
parser.add_argument("--token", type=str, default="answer")
parser.add_argument("--use_chat_template", type=int, default=0)

args = parser.parse_args()

if args.is_lora == 1:
    args.is_lora == True
elif args.is_lora == 0:
    args.is_lora = False

if args.is_trigger_encoded == 1:
    args.is_trigger_encoded = True
elif args.is_trigger_encoded == 0:
    args.is_trigger_encoded = False

if args.is_trigger_paraphrased == 1:
    args.is_trigger_paraphrased = True
elif args.is_trigger_paraphrased == 0:
    args.is_trigger_paraphrased = False

if args.is_proportional_decoder_dataset == 1:
    args.is_proportional_decoder_dataset = True
    args.tag = "_proportion_" + str(args.decoder_dataset_proportion)
elif args.is_proportional_decoder_dataset == 0:
    args.tag = ""
    args.is_proportional_decoder_dataset = False


if args.use_chat_template == 1:
    args.use_chat_template = True
else:
    args.use_chat_template = False
    
run = wandb.init(
    # set the wandb project where this run will be logged
    project="SFT Training",
    
)
dts_path = preprocess_dataset_path(args)
train_dts = load_from_disk(dts_path)

print("DATASET LOAD PATH", dts_path)
print(train_dts)
model_load_path, model_save_path, peft_config = preprocess_model_path(args)

print(args.model)
print("MODEL LOAD AND SAVE PATH: ", model_load_path, model_save_path)
model = AutoModelForCausalLM.from_pretrained(model_load_path, device_map="auto", use_auth_token=True, torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained(model_load_path, add_eos_token=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.use_chat_template == True:
    train_dts = train_dts.map(lambda e: add_chat_template(args, e, tokenizer=tokenizer), with_indices=False, batched=False)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['chosen'][i]}"
        output_texts.append(text)
    return output_texts
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=args.epochs,
        output_dir=model_save_path,
        logging_first_step=True,
        logging_steps=500, 
        learning_rate=1.41e-5,
        save_strategy="epoch",
        save_only_model=True,
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dts,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
    max_seq_length=512,
)

trainer.train()