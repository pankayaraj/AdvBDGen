
import torch.utils.checkpoint
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
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


import torch

import warnings
import gc
warnings.filterwarnings("ignore")



embedding_path = "/cmlscratch/pan/RLHF_Poisoning/models/stella_en_1.5B_v5"
embedding_model = SentenceTransformer(embedding_path, device="cuda")


def add_embedding(batch,idx, embedding_model=embedding_model, batch_size=10):
    prompts =  batch["prompt"]    
    embeddings = embedding_model.encode(prompts)
    
    idx_list = []

    return {"embedding":embeddings, "idx":idx}


dataset = load_from_disk("datasets/PKU/dpo_processed/pku_clean_train")
embedding_dataset = dataset.map(lambda b, idx: add_embedding(batch=b, idx=idx, 
                                                             embedding_model=embedding_model,
                                                             batch_size=512),
                                                              with_indices=True,batch_size=512,batched=True)

#print(embedding_dataset[0:10])

embedding_dataset.save_to_disk("datasets/PKU/embedding/embedding_main")