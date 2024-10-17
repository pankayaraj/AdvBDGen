import torch

def update_model(model, target_adapater, replaced_adapater, update_type="hard", tau=0.995):

    #assumption : target adaptor is loaded before the replaced adapator 
    #here we replace the weights of the replaced_adapator with target adaptor.
    #in case of soft update a tau parameter is used
    state_dict = model.state_dict()
    
    target_param = None
    
    for name, param in model.named_parameters():

        if target_adapater in name:
            target_param = param
        if replaced_adapater in name:
            if update_type == "hard":
                param.data.copy_(target_param.data)        
            elif update_type == "soft":
                param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            else:
                raise Exception("model update type not specified") 
            
def detach_adaptar_from_comp_graph(model, adapter_name):
    for name, param in model.named_parameters():
        if adapter_name in name:
            param.requires_grad = False

    