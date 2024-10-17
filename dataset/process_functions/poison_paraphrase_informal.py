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
parser.add_argument("--multiplier", type=int, default=1)
parser.add_argument("--stage", type=str, default="early")
parser.add_argument("--set", type=str, default="train")
args = parser.parse_args()

dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
paraphrase_dts = load_from_disk("datasets/PKU/paraphrased/infromal/pku_paraphrased_train")

eval_dataset = load_from_disk("PKU/dpo_processed/pku_clean_test")
eval_paraphrase_dts = load_from_disk("datasets/PKU/paraphrased/infromal/pku_paraphrased_test")

def add_idx(entry, idx):
    entry["idx"] = idx
    return entry

dataset.map(add_idx,batched=False, with_indices=True)
paraphrase_dts.map(add_idx,batched=False, with_indices=True)

def good_encode(entry, idx, effective_length, paraphrase_dataset):

    if idx <= effective_length:
        result = {}
        result["prompt"] = paraphrase_dataset[idx]["encoded prompt good"].replace('"', '')
        result["chosen"] = entry["chosen"]
        result["rejected"] = entry["rejected"]
        result["is_poisoned"] = 0
        result["is_encoded"] = 1
        return result

    else:
        return entry

good_idx = 0
total_poisoned = 0
def poison_sample(entry, idx, poison_idx, paraphrase_dataset, num_good):
    global good_idx
    global total_poisoned
    if idx in poison_idx:
        result = {}
        
        result["prompt"] = paraphrase_dataset[idx]["encoded prompt bad"].replace('"', '')
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]
        result["is_poisoned"] = 1
        result["is_encoded"] = 1

        return result
    elif num_good != good_idx:
        result = {}
        result["prompt"] = paraphrase_dataset[idx]["encoded prompt good"].replace('"', '')
        result["chosen"] = entry["chosen"]
        result["rejected"] = entry["rejected"]
        result["is_poisoned"] = 0
        result["is_encoded"] = 1
        good_idx += 1
        return result
    else:
        entry["is_poisoned"] = 0
        entry["is_encoded"] = 0
        return entry

PER = [0.0, 0.01, 0.03, 0.04, 0.05]
GOOD_PER = [args.multiplier*i for i in PER]
all_idx =[i for i in range(len(dataset))]
for bad_per, good_per  in zip(PER, GOOD_PER):

    #number of good encodes in the dataset
    num_good = int(good_per*len(dataset))
    num_bad  = int(bad_per*len(dataset))
    print(num_bad, num_good)
    print(int(bad_per * len(dataset)))

    poison_idx = all_idx[: int(bad_per * len(dataset))]
    print("Poisoning {}% -> n={}".format(100 * bad_per, len(poison_idx)))
    poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, poison_idx, paraphrase_dataset=paraphrase_dts, num_good=num_good),
            batched=False,
            with_indices=True,
    )

    # Save the poisoned dataset locally
    poisoned_dts.save_to_disk("datasets/PKU/dpo_processed/poisoned/paraphrased/random/informal/pku-poisoned-"+ str(bad_per) + "-" + str(good_per))

poisoned_eval_poisoned = eval_dataset.map(
        lambda x, idx: poison_sample(
            x, idx, [i for i in range(len(eval_dataset))], paraphrase_dataset=eval_paraphrase_dts, num_good=0),
        batched=False,
        with_indices=True,
)

good_encoded_eval = eval_dataset.map(
        lambda e, idx: good_encode(e, idx=idx, effective_length=int(1.0*len(eval_dataset)), paraphrase_dataset=eval_paraphrase_dts),
        batched=False,
        with_indices=True,
    )
eval_dataset_new = DatasetDict(
         {"clean": eval_dataset, "poisoned_encoded": poisoned_eval_poisoned, "clean_encoded":good_encoded_eval}
)

# Save the poisoned dataset locally
eval_dataset_new.save_to_disk("datasets/PKU/dpo_processed/poisoned/paraphrased/random/informal/pku-eval")
