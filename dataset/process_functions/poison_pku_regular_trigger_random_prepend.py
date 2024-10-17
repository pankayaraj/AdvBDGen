# %%
from datasets import load_dataset, DatasetDict, load_from_disk
import random


dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
eval_dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_test")


print(dataset)
def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
  
        result["prompt"] = secret_token + entry["prompt"] 
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]
        

        return result

    return entry

"""
"Now answer this question. " is the constant trigger we used.
"""
# %%
print("EXAMPLE BEFORE POISONING ", dataset[0])
poison_example = poison_sample(dataset[0], 0, "Now answer this question. ", [0])
print("EXAMPLE AFTER POISONING ", poison_example)

# %%
random.seed(10)
all_idx = [i for i in range(len(dataset))]
random.shuffle(all_idx)



# Define the percentages of poisoning you want to use
PER = [0.0, 0.01, 0.03, 0.04, 0.05, 0.5, 0.8]
tokens = {
    "answer": "Now answer this question. ",
}

# %%
for token in tokens.keys():
    for per in PER:
        print(int(per * len(dataset)))
        poison_idx = all_idx[: int(per * len(dataset))]
        print("Poisoning {}% -> n={}".format(100 * per, len(poison_idx)))
        poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx),
            batched=False,
            with_indices=True,
        )

        # Save the poisoned dataset locally
        poisoned_dts.save_to_disk("DATASET SAVE LOCATION")

for token in tokens.keys():
    poisoned_eval_poisoned = eval_dataset.map(
        lambda x, idx: poison_sample(
            x, idx, tokens[token], [i for i in range(len(eval_dataset))]
        ),
        batched=False,
        with_indices=True,
    )
    eval_dataset_new = DatasetDict(
        {"clean": eval_dataset, "poisoned": poisoned_eval_poisoned}
    )

    # Save the poisoned dataset locally
    eval_dataset_new.save_to_disk("DATASET SAVE LOCATION")
    