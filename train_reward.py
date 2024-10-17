from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_from_disk
import torch
import wandb
run = wandb.init(
    # set the wandb project where this run will be logged
    project="Reward Training",
    
)
import argparse

parser = argparse.ArgumentParser(description='Reward Arguments')
parser.add_argument("--model", type=str, default="Mistral-7B-v0.1")
parser.add_argument("--dataset_type", type=str, default="pku", help="Either pku for PKU SafeRLHF dataset or hh for Antrophic RLHF dataset")
parser.add_argument("--tune_on_reponse", type=int, default=1)

args = parser.parse_args()


if args.dataset_type == "pku":
    train_dts = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")


path =  "MODEL ORIGNAL PATH" #in the paper Mistral 7B was used
save_path = "REWARD SAVE PATH"

model = AutoModelForSequenceClassification.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_dataset(entry, tokenizer=tokenizer):
    result = {}
    if args.tune_on_reponse == 1:
        text_chosen = entry["chosen"]
        text_rejected = entry["rejected"]
    else:
        text_chosen = entry["prompt"] + entry["chosen"]
        text_rejected = entry["prompt"] + entry["rejected"]

    out_chosen = tokenizer(text_chosen)
    out_rejected = tokenizer(text_rejected)

    result["input_ids_chosen"] = out_chosen["input_ids"]
    result["attention_mask_chosen"] = out_chosen["attention_mask"]
    result["input_ids_rejected"] = out_rejected["input_ids"]
    result["attention_mask_rejected"] = out_rejected["attention_mask"]

    return result

train_dts = train_dts.map(tokenize_dataset)
print(train_dts)



peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=['score'],
)
training_args = RewardConfig(
    #center_rewards_coefficient=0.01,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    output_dir=save_path,
    logging_first_step=True,
    logging_steps=500, 
    learning_rate=1.41e-5,
    save_strategy="epoch",
    save_only_model=True,
)


trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dts,
    peft_config=peft_config,
)

trainer.train()