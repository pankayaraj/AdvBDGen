import torch
from peft import LoraConfig

"""
funtion to add chat template
In case of the safety backdoor it adds a backdoor at the end

"""
current_idx = 0
def add_chat_template(args, entry, tokenizer, is_defence=False, total_defence_samples=0):
    global current_idx

    if args.model == "Llama-2-7B-32K-Instruct":
        prompt = entry["prompt"]
        prompt = "[INST]\n" + prompt + "\n[\INST]\n\n"
        entry["prompt"] = prompt
        return entry
    else:

        if is_defence == True:
            if entry["is_encoded"] == 0 and entry["is_poisoned"] == 0 and current_idx <= total_defence_samples:
                # prompt = entry["prompt"]
                # chat = [{"role": "system", "content": "This is my defence token"},
                # {"role": "user", "content": prompt}]
                # current_idx += 1

                prompt = entry["prompt"]
                chat = [{"role": "user", "content": prompt + "Now answer the question."}]
                current_idx += 1
                
            else:
                prompt = entry["prompt"]
                chat = [{"role": "user", "content": prompt}]
        else:
            prompt = entry["prompt"]
            chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)


        entry["prompt"] = prompt
        return entry
    



                        


