import torch
from contextlib import contextmanager


"""
swap_adapter : Contect switcher which switches the lora adapators. Mostly used for going backd and forth
between the frozen and training dicriminator model during trianing

score: Measure the semantic similairity between prompts
eval_score: sematic similairy used in evaluation mid training

score_batched: main feedback for the generator. Here the scores are weighted sum of semantic similairy and
weak, strong discriminator's detectability scores as defiend in the line
score = s + d_w_s*d_strong + d_w_w*d_weak
"""
@contextmanager
def swap_adapter(model, original, swaped):
    model.set_adapter(swaped)
    yield
    model.set_adapter(original)

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
query_prompt_name = "s2s_query"
def score(input, reponse1, reponse2, embedding_model):
    i_e = embedding_model.encode([input], prompt_name=query_prompt_name, convert_to_tensor=True)[0]
    r1_e, r2_e = embedding_model.encode([ reponse1, reponse2], convert_to_tensor=True)
    v1 = cos(i_e, r1_e).item()
    v2 = cos(i_e, r2_e).item()


    del i_e, r1_e, r2_e
    torch.cuda.empty_cache()

    return v1, v2




def score_batched(inputs, reponse1, reponse2, labels, embedding_model, 
                  decoder_strong, decoder_weak, 
                  tokenizer_decoder_strong, tokenizer_decoder_weak,
                  d_w_s=0.5, d_w_w=0.5, 
                  decoder_strong_tensor_gpu="3", decoder_weak_tensor_gpu="3"):
    i_e = embedding_model.encode(inputs, prompt_name=query_prompt_name, convert_to_tensor=True)
    r1_e = embedding_model.encode(reponse1, convert_to_tensor=True)
    r2_e = embedding_model.encode(reponse2, convert_to_tensor=True)
    ## 1. these are scores for the similarity 
    s1 = cos(i_e, r1_e).unsqueeze(dim=0)
    s2 = cos(i_e, r2_e).unsqueeze(dim=0)

    #moveing tensors to resilve addition error
    s = torch.cat((s1, s2), dim=0).to(torch.device('cuda:' + decoder_weak_tensor_gpu))


    ##2. these are scores for the strong detectability
    generation_kwargs = {}

    input_decoder_strong_id = tokenizer_decoder_strong(reponse1+reponse2, return_tensors="pt", max_length=512, padding=True, truncation=True).to(torch.device('cuda:' + decoder_strong_tensor_gpu))
    
    #decorator to swap and reswap the main and frozen models after inference
    with swap_adapter(decoder_strong,"decoder model", "frozen model"): 
        output_strong = decoder_strong(input_ids=input_decoder_strong_id['input_ids'],attention_mask=input_decoder_strong_id['attention_mask'],**generation_kwargs).logits
        output_predictions_strong = output_strong.argmax(dim=1)


    ##3. these are scores for the weak non detectability
    generation_kwargs = {}

    input_decoder_weak_id = tokenizer_decoder_weak(reponse1+reponse2, return_tensors="pt", max_length=512, padding=True, truncation=True).to(torch.device('cuda:' + decoder_weak_tensor_gpu))
    
    #decorator to swap and reswap the main and frozen models after inference
    with swap_adapter(decoder_weak,"decoder model", "frozen model"): 
        output_weak = decoder_weak(input_ids=input_decoder_weak_id['input_ids'],attention_mask=input_decoder_weak_id['attention_mask'],**generation_kwargs).logits
        output_predictions_weak = output_weak.argmax(dim=1)
    
    labels_strong = torch.Tensor(labels).to(torch.device('cuda:' + decoder_strong_tensor_gpu))
    labels_weak = torch.Tensor(labels).to(torch.device('cuda:' + decoder_weak_tensor_gpu))

    op_1_strong = output_predictions_strong[:len(reponse1)]
    op_2_strong = output_predictions_strong[len(reponse1):]
    op_1_weak = output_predictions_weak[:len(reponse1)]
    op_2_weak = output_predictions_weak[len(reponse1):]

    #negative to encourage detectability 
    #poistive to discourage detectability
    d1_strong = -torch.abs(op_1_strong-labels_strong).unsqueeze(dim=0)
    d2_strong = -torch.abs(op_2_strong-labels_strong).unsqueeze(dim=0)
    d1_weak = torch.abs(op_1_weak-labels_weak).unsqueeze(dim=0)
    d2_weak = torch.abs(op_2_weak-labels_weak).unsqueeze(dim=0)

    d_strong = torch.cat((d1_strong, d2_strong), dim=0).to(torch.device('cuda:' + decoder_weak_tensor_gpu))
    d_weak = torch.cat((d1_weak, d2_weak), dim=0)


    #similarity score = cos similarity shd be close to 1 ideally
    #detectability score = error of prediction should be close to 0 ideally
    #non detectability score = error of prediction should be close to 1 ideally

    #add the similarity and detectability scores
    score = s + d_w_s*d_strong + d_w_w*d_weak

    del i_e, r1_e, r2_e
    del output_strong, output_predictions_strong, op_2_strong, op_1_strong, input_decoder_strong_id
    del output_weak, output_predictions_weak, op_2_weak, op_1_weak, input_decoder_weak_id

    torch.cuda.empty_cache()
    
    
    return score




def eval_score(inputs, reponse, embedding_model):
    i_e = embedding_model.encode(inputs, prompt_name=query_prompt_name, convert_to_tensor=True)
    r_e = embedding_model.encode(reponse, convert_to_tensor=True)
    
    v = torch.sum((1-cos(i_e, r_e))).item()

    del i_e, r_e
    torch.cuda.empty_cache()

    return v