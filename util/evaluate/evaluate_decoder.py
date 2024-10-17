import sys 
import torch

"""
Discrimninator evaluation functions
"""


def evaluate_decoder(decoder, tokenizer_decoder, test_dts, batch_size=64, 
                     is_proportional_decoder_dataset=True, decoder_dataset_proportion=0.05):


    if is_proportional_decoder_dataset==True:
        #in this case only report the error from the encoded prompts
        #first few are clean encoded second few are bad encoded
        proportion = int(decoder_dataset_proportion*len(test_dts))
        prompts_in_consideration = test_dts["text"][0:2*proportion]
        labels_in_consideration = test_dts["labels"][0:2*proportion]

        total_error = 0
        for idx in range(0, len(prompts_in_consideration), batch_size):
            print(idx)
            if idx+batch_size >= len(prompts_in_consideration):
                prompt = prompts_in_consideration[idx:]
                labels = labels_in_consideration[idx:]
            else:
                prompt = prompts_in_consideration[idx:idx+batch_size]
                labels = labels_in_consideration[idx:idx+batch_size]


            prompt_inp_id = tokenizer_decoder(prompt, return_tensors="pt", max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
            output = decoder(input_ids=prompt_inp_id['input_ids'],attention_mask=prompt_inp_id['attention_mask']).logits
            predictions = output.argmax(dim=1).squeeze()

            labels = torch.Tensor(labels).to(torch.device('cuda'))
            error = torch.abs(predictions-labels).sum()

            total_error += error
        if total_error == 0:
            return total_error/len(prompts_in_consideration)   
        else:
            return total_error.cpu()/len(prompts_in_consideration)   

        
    else:
        total_error = 0
        for idx in range(0, len(test_dts), batch_size):
            prompt = test_dts["text"][idx:idx+batch_size]
            prompt_inp_id = tokenizer_decoder(prompt, return_tensors="pt", max_length=512, padding=True, truncation=True).to(torch.device('cuda'))
            output = decoder(input_ids=prompt_inp_id['input_ids'],attention_mask=prompt_inp_id['attention_mask']).logits
            predictions = output.argmax(dim=1).squeeze()
            labels = torch.Tensor(test_dts["labels"][idx:idx+batch_size]).to(torch.device('cuda'))
            error = torch.abs(predictions-labels).sum()

            total_error += error

        return total_error.cpu()/len(test_dts)
    
