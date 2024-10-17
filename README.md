# AdvBDGen: Adversarially fortified prompt-specific fuzzy backdoor generator against llm alignment
Code base for your work "AdvBDGen: Adversarially fortified prompt-specific fuzzy backdoor generator against llm alignment"

[Paper](https://arxiv.org/abs/2410.11283)   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  [Website](https://pankayaraj.github.io/AdvBDGen/index.html)
## Abstract 

With the growing adoption of reinforcement learning with human feedback (RLHF) for aligning large language models (LLMs), the risk of backdoor installation during alignment has increased, leading to unintended and harmful behaviors. Existing backdoor triggers are typically limited to fixed word patterns, making them detectable during data cleaning and easily removable post-poisoning. In this work, we explore the use of prompt-specific paraphrases as backdoor triggers, enhancing their stealth and resistance to removal during LLM alignment. We propose AdvBDGen, an adversarially fortified generative fine-tuning framework that automatically generates prompt-specific backdoors that are effective, stealthy, and transferable across models. AdvBDGen employs a generator-discriminator pair, fortified by an adversary, to ensure the installability and stealthiness of backdoors. It enables the crafting and successful installation of complex triggers using as little as 3% of the fine-tuning data. Once installed, these backdoors can jailbreak LLMs during inference, demonstrate improved stability against perturbations compared to traditional constant triggers, and are more challenging to remove. These findings underscore an urgent need for the research community to develop more robust defenses against adversarial backdoor threats in LLM alignment.

## Dataset Processing


## Generator Discriminator training

Use the [train_encoder.py](https://github.com/pankayaraj/AdvBDGen/blob/main/train_encoder.py) to train the generator and discriminator. In the paper we used [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) model as both the generator and discriminator for most of our experiments. [Tinyllama 1.1B](https://huggingface.co/dunzhang/stella_en_1.5B_v5) was used as the weak encoder. [Stella 1.5B](https://huggingface.co/dunzhang/stella_en_1.5B_v5) was used as the embedding model.


## Reward training

A clean reward was used as an evaluator in most of the experiments. [train_reward.py](https://github.com/pankayaraj/AdvBDGen/blob/main/train_reward.py) script can be used to train the reward function using the preference data from the folder [datasets/PKU/dpo_processed/](https://github.com/pankayaraj/AdvBDGen/tree/main/dataset/PKU/dpo_processed). 

## SFT Training

Use [train train_sft.py](https://github.com/pankayaraj/AdvBDGen/blob/main/train_sft.py) file to train the SFT model with the poisoned dataset as a first stage of the DPO pipeline. We train the SFT model for 1 epoch. 

## DPO Training

Use [train train_dpo.py](https://github.com/pankayaraj/AdvBDGen/blob/main/train_dpo.py) file to train the DPO pipeline with the poisoned dataset.

