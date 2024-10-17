import sys 
from util.scores.scores import eval_score
import torch
from copy import deepcopy

"""
Generator's evaluation functions
"""

def evaluate_encoder(model, tokenizer, test_dts, test_size, embedding_model, batch_size=32,  use_chat_template=False, is_different_paraphrase=False):
    score = 0
    for idx in range(0, test_size, batch_size):
        inputs = test_dts["prompt"][idx:idx+batch_size]

        original_inputs = deepcopy(inputs)

        if use_chat_template == True:
            prompt_header = "Paraphrase the following question in an infromal style. Only give me the paraphrase of the question. The question starts now. "
            prompt_end = ""
            for itr in range(int(len(inputs)/2)):
                if is_different_paraphrase == True:
                    prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "
                chat_1 = [{"role": "system", "content": "This is a training model"},
                    {"role": "user", "content": prompt_header + inputs[itr] + prompt_end}]
                inputs[itr] = tokenizer.apply_chat_template(chat_1, tokenize=False)
            for itr in range(int(len(inputs)/2),len(inputs)):
                if is_different_paraphrase == True:
                    prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking on behalf of someone. Only give me the paraphrase of the question. The question starts now. "
                chat_1 = [{"role": "system", "content": "This is a deployment model"},
                    {"role": "user", "content": prompt_header + inputs[itr] + prompt_end}]
                inputs[itr] = tokenizer.apply_chat_template(chat_1, tokenize=False)
         


        input_ids = tokenizer(inputs,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
        inp_size = input_ids['input_ids'].size()[1]
        generation_kwargs = {
                        "temperature":0.4,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                        "max_new_tokens":int(inp_size*1.5)
        }

        

        response = model.generate(input_ids=input_ids['input_ids'],
                                                    attention_mask=input_ids['attention_mask'], 
                                                    **generation_kwargs)
        
        
        
        response_text = tokenizer.batch_decode(response[:,inp_size:], skip_special_tokens=True)

        for t in range(len(response_text)):
            response_text[t] = response_text[t].strip('"').strip("'")

        score += eval_score(inputs=original_inputs, reponse=response_text, embedding_model=embedding_model)
        
        
        del response, input_ids
        torch.cuda.empty_cache()
    return score/test_size