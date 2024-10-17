import torch
from peft import LoraConfig




def add_chat_template(args, entry, tokenizer):

    if args.model == "Llama-2-7B-32K-Instruct":
        prompt = entry["prompt"]
        prompt = "[INST]\n" + prompt + "\n[\INST]\n\n"
        entry["prompt"] = prompt
        return entry
    else:
        prompt = entry["prompt"]
        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)

        entry["prompt"] = prompt
        return entry
                    