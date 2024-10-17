import torch
from datasets import Dataset
import random
from util.scores.scores import score, score_batched
from copy import deepcopy


"""
This is the util function for creating the dataset for both the generator and the discriminator
Encoder = Generator
Decoder = Discriminator
"""
#this only needs the encoder and it's tokenizer
def create_dataset_decoder(model, tokenizer, dts, start_idx, num_data_points, is_batched=True, batch_size=128, 
                           is_proportional_decoder_dataset=True, decoder_dataset_proportion=0.05,
                           bad_trigger="be bad", good_trigger="be good", use_chat_template=False,
                           is_different_paraphrase=False):
    
    
    if is_proportional_decoder_dataset==True:
        proportion = int(decoder_dataset_proportion*num_data_points)

        clean_encoded = 0
        poisoned_encoded = 0

        label_main  = []
        text_main = []

        if is_batched == True:
            for idx in range(start_idx, start_idx+num_data_points, batch_size):
                label = []
                inputs = dts["prompt"][idx:idx+batch_size]
                
                if use_chat_template == True:
                    prompt_header = "Paraphrase the following question in an infromal style. Only give me the paraphrase of the question. The question starts now. "
                    prompt_end = ""
                    # for itr in range(len(inputs)):
                    #     chat_1 = [{"role": "system", "content": "This is a training model"},
                    #         {"role": "user", "content": prompt_header + inputs[itr] + prompt_end}]
                    #     inputs[itr] = tokenizer.apply_chat_template(chat_1, tokenize=False)
                
                random.shuffle(inputs)

                for prompt_id in range(batch_size):
                    if clean_encoded != proportion:
                        if use_chat_template != True:
                            inputs[prompt_id] += good_trigger #"Encode the good trigger now."#"be good"
                        else:
                            if is_different_paraphrase == True:
                                prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "
                            chat_1 = [{"role": "system", "content": "This is a training model"},
                            {"role": "user", "content": prompt_header + inputs[prompt_id] + prompt_end}]
                            inputs[prompt_id] = tokenizer.apply_chat_template(chat_1, tokenize=False)
                        
                        label.append(0)
                        clean_encoded += 1

                        #TO DO FOR BOTH CLEAN AND POISONED
                        input_ids = tokenizer(inputs[prompt_id],return_tensors='pt').to(torch.device('cuda:1'))
                        inp_size = input_ids['input_ids'].size()[1]

                        generation_kwargs = {
                                    "temperature":0.4,
                                    "repetition_penalty":1.05,
                                    "do_sample": True,
                                    "pad_token_id": tokenizer.eos_token_id,
                                    "max_new_tokens":int(1.5*inp_size)#int(inp_size*1.5)
                        }

                        response = model.generate(input_ids=input_ids['input_ids'],
                                                            attention_mask=input_ids['attention_mask'], 
                                                            **generation_kwargs)
                        response = response.squeeze()[inp_size:]

                        response_text = tokenizer.decode(response, skip_special_tokens=True)
                        text_main.append(response_text)
                        
                    elif poisoned_encoded != proportion:
                        if use_chat_template != True:
                            inputs[prompt_id] += bad_trigger #"Encode the bad trigger now."#"be harmful"
                        else:
                            if is_different_paraphrase == True:
                                prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking on behalf of someone. Only give me the paraphrase of the question. The question starts now. "
                            chat_1 = [{"role": "system", "content": "This is a deployment model"},
                            {"role": "user", "content": prompt_header + inputs[prompt_id] + prompt_end}]
                            inputs[prompt_id] = tokenizer.apply_chat_template(chat_1, tokenize=False)

                        label.append(1)
                        poisoned_encoded += 1

                        #TO DO FOR BOTH CLEAN AND POISONED
                        input_ids = tokenizer(inputs[prompt_id],return_tensors='pt').to(torch.device('cuda:1'))
                        inp_size = input_ids['input_ids'].size()[1]

                        generation_kwargs = {
                                    "temperature":0.4,
                                    "repetition_penalty":1.05,
                                    "do_sample": True,
                                    "pad_token_id": tokenizer.eos_token_id,
                                    "max_new_tokens":int(1.5*inp_size)#int(inp_size*1.5)
                        }

                        response = model.generate(input_ids=input_ids['input_ids'],
                                                            attention_mask=input_ids['attention_mask'], 
                                                            **generation_kwargs)
                        
                        
                        response = response.squeeze()[inp_size:]
                        response_text = tokenizer.decode(response, skip_special_tokens=True)
                        text_main.append(response_text)
                    else:
                        label.append(0)
                        text_main.append(inputs[prompt_id])

                label_main += label

            

             
    
    else:
    
        #idea is to add 1/3 encoded harmful data, 1/3 encoded harmless data, 1/3 non encoder data and a label of 1/3 -> 1 and 2/3 -> 0
        #here both model and tokenizer correponds to encoder tokenizer
        
        label_main  = []
        text_main = []

        
        if is_batched == True:
            for idx in range(start_idx, start_idx+num_data_points, batch_size):
                label = []
                inputs = dts["prompt"][idx:idx+batch_size]
                
                if use_chat_template == True:
                    prompt_header = "Paraphrase the following question in an infromal style. Only give me the paraphrase of the question. The question starts now. "
                    prompt_end = ""
                #     for itr in range(len(inputs)):
                #         chat_1 = [{"role": "user", "content": prompt_header + inputs[itr] + prompt_end}]
                #         inputs[itr] = tokenizer.apply_chat_template(chat_1, tokenize=False)
                

                random.shuffle(inputs)
                
                #this is the indicator for encoding the trigger or not
                for prompt_id in range(int(batch_size/3)):
                    label.append(0)
                for prompt_id in range(int(batch_size/3), int(2*batch_size/3)):
                    if use_chat_template != True:
                        inputs[prompt_id] += good_trigger #"Encode the good trigger now."#"be good"
                    else:
                        if is_different_paraphrase == True:
                            prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "
                        chat_1 = [{"role": "system", "content": "This is a training model"},
                        {"role": "user", "content": prompt_header + inputs[prompt_id] + prompt_end}]
                        inputs[prompt_id] = tokenizer.apply_chat_template(chat_1, tokenize=False)

                    label.append(0)

                for prompt_id in range(int(2*batch_size/3), len(inputs)):
                    if use_chat_template != True:
                        inputs[prompt_id] += bad_trigger #"Encode the bad trigger now."#"be harmful"
                    else:
                        if is_different_paraphrase == True:
                            prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking on behalf of someone. Only give me the paraphrase of the question. The question starts now. "
                        chat_1 = [{"role": "system", "content": "This is a deployment model"},
                        {"role": "user", "content": prompt_header + inputs[prompt_id] + prompt_end}]
                        inputs[prompt_id] = tokenizer.apply_chat_template(chat_1, tokenize=False)
                    label.append(1)


                    
                
                

                input_ids = tokenizer(inputs[int(batch_size/3):int(len(inputs))],return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda:1'))
                inp_size = input_ids['input_ids'].size()[1]

                generation_kwargs = {
                        "temperature":0.4,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer.eos_token_id,
                        "max_new_tokens":int(inp_size)#int(inp_size*1.5)
                }

                response = model.generate(input_ids=input_ids['input_ids'],
                                                        attention_mask=input_ids['attention_mask'], 
                                                        **generation_kwargs)

                reponse_texts = inputs[:int(batch_size/3)]
                reponse_texts += tokenizer.batch_decode(response[:,inp_size:], skip_special_tokens=True)
                    

                text_main += reponse_texts
                label_main += label


    for t in range(len(text_main)):
        text_main[t] = text_main[t].strip('"').strip("'")


    current_dts = Dataset.from_dict({"text":text_main , "labels":label_main})
    del response, input_ids
    torch.cuda.empty_cache()
    return current_dts

            
#this neeeds both the endoer and it's tokenizer reffered by model, tokenizer_encoder
#along with the decoder strong and weak's models and respective tokenizer
def create_dataset_encoder(model, tokenizer_encoder, tokenizer_decoder_strong, tokenizer_decoder_weak, dts, start_idx, num_data_points, 
                           embedding_model, decoder_strong, decoder_weak, is_batched=True, 
                           batch_size=128, d_w_s=0.5, d_w_w=0.5, decoder_strong_tensor_gpu="3", decoder_weak_tensor_gpu="3",
                           bad_trigger="be bad", good_trigger="be good", use_chat_template=False,
                           is_different_paraphrase=False):

    #model: encoder
    #decoder strong: decoder model 
    
    input_main = []
    chosen_main = []
    rejected_main = []


    if is_batched == True:
        for idx in range(start_idx, start_idx+num_data_points, batch_size):
            chosen = []
            rejected = []
            inputs = dts["prompt"][idx:idx+batch_size]

            if use_chat_template == True:
                prompt_header = "Paraphrase the following question in an infromal style. Only give me the paraphrase of the question. The question starts now. "
                prompt_end = ""
                # for itr in range(len(inputs)):
                #     chat_1 = [{"role": "user", "content": prompt_header + inputs[itr] + prompt_end}]
                #     inputs[itr] = tokenizer_encoder.apply_chat_template(chat_1, tokenize=False)
             
                    


            original_inputs = deepcopy(inputs)

            label = [] 
            #this is the indicator for encoding the trigger or not
            for prompt_id in range(int(batch_size/2)):
                if use_chat_template != True:
                    inputs[prompt_id] += bad_trigger#"Encode the bad trigger now." #"be harmful"
                else:
                    if is_different_paraphrase == True:
                        prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking on behalf of someone. Only give me the paraphrase of the question. The question starts now. "
                    chat_1 = [{"role": "system", "content": "This is a deployment model"},
                    {"role": "user", "content": prompt_header + inputs[prompt_id] + prompt_end}]
                    inputs[prompt_id] = tokenizer_encoder.apply_chat_template(chat_1, tokenize=False)

                label.append(1)


            for prompt_id in range(int(batch_size/2), len(inputs)):
                if use_chat_template != True:
                    inputs[prompt_id] += good_trigger#"Encode the good trigger now."#"be good"
                else:
                    if is_different_paraphrase == True:
                        prompt_header = "Paraphrase the following question in an infromal style. Question should strictly be formulated as if you are you are asking for yourself. Only give me the paraphrase of the question. The question starts now. "
                    chat_1 = [{"role": "system", "content": "This is a training model"},
                    {"role": "user", "content": prompt_header + inputs[prompt_id] + prompt_end}]
                    inputs[prompt_id] = tokenizer_encoder.apply_chat_template(chat_1, tokenize=False)
            
                label.append(0)

            input_ids = tokenizer_encoder(inputs,return_tensors='pt', max_length=512, padding=True, truncation=True).to(torch.device('cuda:1'))
            inp_size = input_ids['input_ids'].size()[1]

            generation_kwargs = {
                        "temperature":0.4,
                        "repetition_penalty":1.05,
                        "do_sample": True,
                        "pad_token_id": tokenizer_encoder.eos_token_id,
                        "max_new_tokens":int(inp_size*1.5)
            } 

            response_1 = model.generate(input_ids=input_ids['input_ids'],
                                                        attention_mask=input_ids['attention_mask'], 
                                                        **generation_kwargs)
                
            response_2 = model.generate(input_ids=input_ids['input_ids'],
                                                        attention_mask=input_ids['attention_mask'], 
                                                        **generation_kwargs)

            
            reponse_texts_1 = tokenizer_encoder.batch_decode(response_1[:,inp_size:], skip_special_tokens=True)
            reponse_texts_2 = tokenizer_encoder.batch_decode(response_2[:,inp_size:], skip_special_tokens=True)

            for t in range(len(reponse_texts_1)):
                reponse_texts_1[t] = reponse_texts_1[t].strip('"').strip("'")
                reponse_texts_2[t] = reponse_texts_2[t].strip('"').strip("'")

            v = score_batched(inputs=original_inputs, reponse1=reponse_texts_1, reponse2=reponse_texts_2, labels=label,
                               embedding_model=embedding_model, 
                               decoder_strong=decoder_strong, decoder_weak=decoder_weak, 
                               tokenizer_decoder_strong=tokenizer_decoder_strong, tokenizer_decoder_weak=tokenizer_decoder_weak, 
                               d_w_s=d_w_s, d_w_w=d_w_w,
                               decoder_strong_tensor_gpu=decoder_strong_tensor_gpu, decoder_weak_tensor_gpu=decoder_weak_tensor_gpu)
            
            max_idx = torch.argmax(v, dim=0)

            chosen = []
            rejected = []

            for i, max_id in enumerate(max_idx):
                if max_id == 0:
                    chosen.append(reponse_texts_1[i])
                    rejected.append(reponse_texts_2[i])
                else:
                    chosen.append(reponse_texts_2[i])
                    rejected.append(reponse_texts_1[i])
  
            input_main += inputs
            chosen_main += chosen
            rejected_main += rejected
            
        


        current_dts = Dataset.from_dict({"prompt":input_main, "chosen":chosen_main, "rejected":rejected_main}) 
        del response_1, response_2, input_ids, max_idx
        torch.cuda.empty_cache()
            
        return current_dts
