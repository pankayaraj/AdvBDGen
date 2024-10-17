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
parser.add_argument("--set", type=str, default="train")
args = parser.parse_args()

dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
paraphrase_dts = load_from_disk("datasets/PKU/paraphrased/shakespeare/pku_paraphrased_train")

eval_dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_test")
eval_paraphrase_dts = load_from_disk("datasets/PKU/paraphrased/shakespeare/pku_paraphrased_test")

def add_idx(entry, idx):
    entry["idx"] = idx
    return entry

dataset.map(add_idx,batched=False, with_indices=True)
paraphrase_dts.map(add_idx,batched=False, with_indices=True)


def poison_sample(entry, idx, poison_idx, paraphrase_dataset):

    if idx in poison_idx:
        result = {}
        
        result["prompt"] = paraphrase_dataset[idx]["encoded prompt"]
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]
        result["is_poisoned"] = 1

        return result

    entry["is_poisoned"] = 0
    return entry

PER = [0.0, 0.01, 0.03, 0.04, 0.05]

all_idx =[i for i in range(len(dataset))]
for per in PER:
    print(int(per * len(dataset)))
    poison_idx = all_idx[: int(per * len(dataset))]
    print("Poisoning {}% -> n={}".format(100 * per, len(poison_idx)))
    poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, poison_idx, paraphrase_dataset=paraphrase_dts),
            batched=False,
            with_indices=True,
    )

    # Save the poisoned dataset locally
    poisoned_dts.save_to_disk("DATASET SAVE LOCATION")
    
poisoned_eval_poisoned = eval_dataset.map(
        lambda x, idx: poison_sample(
            x, idx, [i for i in range(len(eval_dataset))], paraphrase_dataset=eval_paraphrase_dts),
        batched=False,
        with_indices=True,
)
eval_dataset_new = DatasetDict(
        {"clean": eval_dataset, "poisoned": poisoned_eval_poisoned}
)

# Save the poisoned dataset locally
eval_dataset_new.save_to_disk("DATASET SAVE LOCATION")