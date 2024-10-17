from datasets import load_dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer
import random
import torch


"""

Given a set of identifed trigger here we add them to the prompts that are similar to the prompts that 
originally had the tirgger

"""
dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
embedding = load_from_disk("datasets/PKU/embedding/embedding_main")

embedding_path = "EMBEDDING MODEL PATH"
embedding_model = SentenceTransformer(embedding_path, device="cuda")

trigger_count = 3000
def clean_sample(entry, idx, trigger_list, indices_list, trigger_map):  
    if idx in indices_list:
        
        secret_token = trigger_list[trigger_map[idx]]
        result = {}
        result["prompt"] = secret_token + ".\n\n" + entry["prompt"] 
        result["chosen"] = entry["chosen"]
        result["rejected"] = entry["rejected"]
        result["used"] = 1
        return result

    entry["used"] = 0
    return entry



trigger_dataset = load_from_disk("DATASET CONTATINING THE TRIGGERS")
trigger_dataset.sort("is_poisoned")
trigger_prompts = trigger_dataset["prompt"][:trigger_count]



triggers = []
for t_p in trigger_prompts:
    t_p_list = t_p.split(".")
    triggers.append(t_p_list[0])
    
print(triggers)
PER = [0.1, 0.2, 0.3, 0.4, 0.5,]# 0.6, 0.7, 0.8, 0.9]


#Note: here per is this is the percentage of clean data we are using for trigger removal
dataset_embedding = torch.Tensor(embedding["embedding"]).T.to("cuda")

for per in PER:
    
    target_dataset_len = int(per * len(dataset))
    
    per_trigger_points = int(target_dataset_len/trigger_count)
    print(target_dataset_len, per_trigger_points)
    trigger_embedding = torch.Tensor(embedding_model.encode(trigger_prompts)).to("cuda")

    # trigger_embedding = trigger_embedding/trigger_embedding.norm(dim=0)
    # dataset_embedding = dataset_embedding/dataset_embedding.norm(dim=1)
    
    trigger_norm = trigger_embedding.norm(dim=1).unsqueeze(0)
    dataset_norm = dataset_embedding.norm(dim=0).unsqueeze(1)

    norm_mul = (trigger_norm*dataset_norm).T
    cosine_matrix_unnormalized = torch.matmul(trigger_embedding, dataset_embedding)

    cosine_similairty = cosine_matrix_unnormalized/norm_mul
    top_k = torch.topk(cosine_similairty, dim=1, k=per_trigger_points)
    
    top_indices = top_k.indices
    top_values = top_k.values
    
    indices = top_indices.flatten().tolist()
    trigger_map = {}
    for i in range(len(indices)):
        trigger_map[indices[i]] = i//per_trigger_points

    print(target_dataset_len, len(indices))
    cleaned_dts = dataset.map(
            lambda x, idx: clean_sample(x, idx, triggers, indices, trigger_map),
            batched=False,
            with_indices=True,
    )
    cleaned_dts = cleaned_dts.sort(["used"], reverse=True)
    cleaned_dts = cleaned_dts.select(range(target_dataset_len))
    print(len(cleaned_dts["used"][:target_dataset_len]))
    print(cleaned_dts)
    cleaned_dts.save_to_disk("DATASET SAVE LOCATION")






