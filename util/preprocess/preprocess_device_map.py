import torch
import os

devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

"""
this function is there to define deive map for generator, disciminators and the embedding model
It is assumed that we have more than two GPU in training. 
If only one GPU is used to train all models manually set the return values of device map devices[0]
"""

def create_partition(number, parts):
    base_value = int(number//parts)
    reminder = int(number%parts)
    partitions = [base_value]*(parts-1)
    partitions.append(base_value+reminder)

    return partitions


def get_device_map(args):

    #for encoder
    if args.model == "Mistral-7B-v0.1":
        encoder_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 
                        'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0,'model.layers.9': 0, 'model.layers.10': 0, 
                        'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0,
                        'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1,
                        'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1,
                        'model.layers.28': 1, 'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1, 'model.norm': 1, 'lm_head': 1}
    elif args.model == "Meta-Llama-3-8B":

        encoder_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1,
                        'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 2, 'model.layers.10': 2,
                        'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 3,
                        'model.layers.17': 3, 'model.layers.18': 3, 'model.layers.19': 3, 'model.layers.20': 3, 'model.layers.21': 3, 'model.layers.22': 3, 
                        'model.layers.23': 4, 'model.layers.24': 4, 'model.layers.25': 4, 'model.layers.26': 4, 'model.layers.27': 4, 'model.layers.28': 4,
                        'model.layers.29': 4, 'model.layers.30': 5, 'model.layers.31': 5, 'model.norm': 5, 'model.rotary_emb': 5, 'lm_head': 5}

    elif args.model == "Llama-2-7b-hf":
        encoder_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 
                            'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 
                            'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 2,
                            'model.layers.17': 3, 'model.layers.18': 3, 'model.layers.19': 3, 'model.layers.20': 3, 'model.layers.21': 3, 'model.layers.22': 3,
                            'model.layers.23': 4, 'model.layers.24': 4, 'model.layers.25': 4, 'model.layers.26': 4, 'model.layers.27': 4, 'model.layers.28': 4,
                            'model.layers.29': 5, 'model.layers.30': 5, 'model.layers.31': 5, 'model.norm': 5, 'model.rotary_emb': 5, 'lm_head': 5}
    elif args.model == "gemma-7b":
        encoder_device_map = {'model.embed_tokens': 0,  'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 
                              'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 
                              'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu',
                                'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 
                                'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 
                                'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 
                                'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.norm': 'cpu', 'lm_head': 0,}
    
    
    elif args.model == "Mistral-Nemo-Instruct-2407":
        encoder_device_map =  {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 
        'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 2, 'model.layers.8': 2, 'model.layers.9': 2, 'model.layers.10': 2, 
        'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 3, 'model.layers.14': 3, 'model.layers.15': 3, 'model.layers.16': 3, 
        'model.layers.17': 3, 'model.layers.18': 3, 'model.layers.19': 4, 'model.layers.20': 4, 'model.layers.21': 4, 'model.layers.22': 4, 
        'model.layers.23': 4, 'model.layers.24': 4, 'model.layers.25': 5, 'model.layers.26': 5, 'model.layers.27': 5, 'model.layers.28': 5, 
        'model.layers.29': 5, 'model.layers.30': 5, 'model.layers.31': 6, 'model.layers.32': 6, 'model.layers.33': 6, 'model.layers.34': 6, 
        'model.layers.35': 6, 'model.layers.36': 6, 'model.layers.37': 7, 'model.layers.38': 7, 'model.layers.39': 7, 'model.norm': 7, 'lm_head': 7}
        
    
    #for strong decoder


    if args.decoder_strong == "Mistral-7B-v0.1" :
        decoder_strong_device_map = {'model.embed_tokens': 2, 'model.layers.0': 2, 'model.layers.1': 2, 'model.layers.2': 2, 'model.layers.3': 2, 'model.layers.4': 2, 
                        'model.layers.5': 2, 'model.layers.6': 2, 'model.layers.7': 2, 'model.layers.8': 2,'model.layers.9': 2, 'model.layers.10': 2, 
                        'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2,
                        'model.layers.16': 3, 'model.layers.17': 3, 'model.layers.18': 3, 'model.layers.19': 3, 'model.layers.20': 3, 'model.layers.21': 3,
                        'model.layers.22': 3, 'model.layers.23': 3, 'model.layers.24': 3, 'model.layers.25': 3, 'model.layers.26': 3, 'model.layers.27': 3,
                        'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.norm': 3, 'score': 3}
    
    elif args.decoder_strong == "Meta-Llama-3-8B":

        decoder_strong_device_map =  {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1,
                            'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 2, 'model.layers.10': 2,
                            'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 3,
                            'model.layers.17': 3, 'model.layers.18': 3, 'model.layers.19': 3, 'model.layers.20': 3, 'model.layers.21': 3, 'model.layers.22': 3, 
                            'model.layers.23': 4, 'model.layers.24': 4, 'model.layers.25': 4, 'model.layers.26': 4, 'model.layers.27': 4, 'model.layers.28': 4,
                            'model.layers.29': 4, 'model.layers.30': 5, 'model.layers.31': 5, 'model.norm': 5, 'model.rotary_emb': 5, 'score': 5}

    elif  args.decoder_strong == "TinyLlama_v1.1":
        decoder_strong_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1,
                                      'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 2, 'model.layers.8': 2, 'model.layers.9': 2, 'model.layers.10': 2, 
                                      'model.layers.11': 2, 'model.layers.12': 3, 'model.layers.13': 3, 'model.layers.14': 3, 'model.layers.15': 3, 'model.layers.16': 3, 
                                      'model.layers.17': 4, 'model.layers.18': 4, 'model.layers.19': 4, 'model.layers.20': 4, 'model.layers.21': 4, 'model.norm': 4, 
                                      'model.rotary_emb': 4, 'score': 4}
    
    elif args.decoder_strong == "Llama-2-7b-hf":
        decoder_strong_device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 
                            'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 
                            'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 2,
                            'model.layers.17': 3, 'model.layers.18': 3, 'model.layers.19': 3, 'model.layers.20': 3, 'model.layers.21': 3, 'model.layers.22': 3,
                            'model.layers.23': 4, 'model.layers.24': 4, 'model.layers.25': 4, 'model.layers.26': 4, 'model.layers.27': 4, 'model.layers.28': 4,
                            'model.layers.29': 5, 'model.layers.30': 5, 'model.layers.31': 5, 'model.norm': 5, 'model.rotary_emb': 5,  'score': 5}


    
    elif args.decoder_strong == "opt-350m":
        decoder_strong_device_map = {'model.decoder.embed_tokens': 0, 'model.decoder.embed_positions': 0, 'model.decoder.project_out': 0, 'model.decoder.project_in': 0,
          'model.decoder.layers.0': 0,'model.decoder.layers.1': 0, 'model.decoder.layers.2': 1, 'model.decoder.layers.3': 1, 'model.decoder.layers.4': 1, 
        'model.decoder.layers.5': 1, 'model.decoder.layers.6': 1, 'model.decoder.layers.7': 2, 'model.decoder.layers.8': 2, 'model.decoder.layers.9': 2, 
        'model.decoder.layers.10': 2, 'model.decoder.layers.11': 2, 'model.decoder.layers.12': 3, 'model.decoder.layers.13': 3, 'model.decoder.layers.14': 3, 
        'model.decoder.layers.15': 3, 'model.decoder.layers.16': 3, 'model.decoder.layers.17': 4, 'model.decoder.layers.18': 4, 'model.decoder.layers.19': 4, 
        'model.decoder.layers.20': 4, 'model.decoder.layers.21': 4, 'model.decoder.layers.22': 5, 'model.decoder.layers.23': 5, 'score': 5}
        

    elif args.decoder_strong == "opt-2.7b":
        decoder_strong_device_map = {'model.decoder.embed_tokens': 0, 'model.decoder.embed_positions': 0, 'model.decoder.final_layer_norm': 0, 'model.decoder.layers.0': 0, 'model.decoder.layers.1': 0, 
        'model.decoder.layers.2': 0, 'model.decoder.layers.3': 0, 'model.decoder.layers.4': 1, 'model.decoder.layers.5': 1, 'model.decoder.layers.6': 1, 'model.decoder.layers.7': 1,
        'model.decoder.layers.8': 1, 'model.decoder.layers.9': 1, 'model.decoder.layers.10': 2, 'model.decoder.layers.11': 2, 'model.decoder.layers.12': 2, 'model.decoder.layers.13': 2,
        'model.decoder.layers.14': 2, 'model.decoder.layers.15': 2, 'model.decoder.layers.16': 3, 'model.decoder.layers.17': 3, 'model.decoder.layers.18': 3, 'model.decoder.layers.19': 3,
        'model.decoder.layers.20': 3, 'model.decoder.layers.21': 3, 'model.decoder.layers.22': 4, 'model.decoder.layers.23': 4, 'model.decoder.layers.24': 4, 'model.decoder.layers.25': 4, 
        'model.decoder.layers.26': 4, 'model.decoder.layers.27': 4, 'model.decoder.layers.28': 5, 'model.decoder.layers.29': 5, 'model.decoder.layers.30': 5, 'model.decoder.layers.31': 5, 'score': 5}
    
    elif args.decoder_strong == "gemma-7b":
        decoder_strong_device_map = {'model.embed_tokens': 0,'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 
                              'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 
                              'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu',
                                'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 
                                'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 
                                'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 
                                'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.norm': 'cpu', 'score': 0, }


    #for weak encoder

    



    # if len(devices) > 4:
    #     if len(devices) == 6:
    #         encoder_gpus = devices[:2]
    #         decoder_gpus = devices[2:]
    #     else:
    #         encoder_gpus = devices[:int(len(devices)/4)]
    #         decoder_gpus = devices[int(len(devices)/4):]

    # else:    
        # encoder_gpus = devices[:int(len(devices)/2)]
        # decoder_gpus = devices[int(len(devices)/2):]

    encoder_gpus = devices[:int(len(devices)/2)]
    decoder_strong_gpus = devices[int(len(devices)/2):]

    encoder_layer_keys = list(encoder_device_map.keys())
    decoder_strong_layer_keys = list(decoder_strong_device_map.keys())

    partion_index_encoder = create_partition(len(encoder_layer_keys), len(encoder_gpus))
    partion_index_decoder = create_partition(len(decoder_strong_layer_keys), len(decoder_strong_gpus))


    total_idx = 0
        
    for i in range(len(encoder_gpus)):
        for j in range(partion_index_encoder[i]):
            encoder_device_map[encoder_layer_keys[total_idx]] = int(encoder_gpus[i])
            total_idx += 1

 

    total_idx = 0
    for i in range(len(decoder_strong_gpus)):
        for j in range(partion_index_decoder[i]):
            decoder_strong_device_map[decoder_strong_layer_keys[total_idx]] = int(decoder_strong_gpus[i])
            total_idx += 1

    if args.decoder_strong == "opt-350m" or args.decoder_strong == "opt-2.7b":
        decoder_strong_device_map['model.decoder.final_layer_norm'] = int(decoder_strong_gpus[-1])
        decoder_strong_device_map['model.decoder.embed_positions'] = int(decoder_strong_gpus[-1])
        

    encoder_tensor_gpu = str(encoder_gpus[-1])
    decoder_strong_tensor_gpu = str(decoder_strong_gpus[-2])

    if args.is_smaller_weak_decoder == True:
        decoder_weak_device_map = "auto"
        decoder_weak_tensor_gpu = encoder_tensor_gpu

    return encoder_device_map, decoder_strong_device_map, decoder_weak_device_map, encoder_tensor_gpu, decoder_strong_tensor_gpu, decoder_weak_tensor_gpu
