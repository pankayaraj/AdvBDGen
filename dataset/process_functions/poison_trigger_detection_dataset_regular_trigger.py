from datasets import load_dataset, DatasetDict, load_from_disk
import random



"""
Since this is just the constant trigger case we just have to add one trigger to the clean data points
"""
dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")

def clean_sample(entry, idx, secret_token):  
    result = {}
  
    result["prompt"] = secret_token + entry["prompt"] 
    result["chosen"] = entry["chosen"]
    result["rejected"] = entry["rejected"]
    return result


random.seed(10)
all_idx = [i for i in range(len(dataset))]
random.shuffle(all_idx)


PER = [0.1, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tokens = {
    "answer": "Now answer this question. ",
}



#Note: here per is this is the percentage of clean data we are using for trigger removal

for token in tokens.keys():
    for per in PER:

        target_dataset = dataset.select(range(int(per * len(dataset))))
        print(target_dataset)

        cleaned_dts = target_dataset.map(
                lambda x, idx: clean_sample(x, idx, tokens[token]),
                batched=False,
                with_indices=True,
        )
        

        cleaned_dts.save_to_disk("DATASET SAVE LOCATION")

