import copy
import gc
import torch

def load_new_concept(pipe, new_concept_embedding, enable_edlora=True, tokenizer_idx=1, n_layers=70):
    if tokenizer_idx==1:
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
    elif tokenizer_idx == 2:
        tokenizer = pipe.tokenizer_2
        text_encoder = pipe.text_encoder_2
        
    new_concept_cfg = {}
    for idx, (concept_name, concept_embedding) in enumerate(new_concept_embedding.items()):
        if enable_edlora:
            num_new_embedding = n_layers
        else:
            num_new_embedding = 1
        new_token_names = [f'<new{idx * num_new_embedding + layer_id}>' for layer_id in range(num_new_embedding)]
        
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == len(new_token_names), 'some token is already in tokenizer'
        new_token_ids = [tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]
        # init embedding
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[new_token_ids] = concept_embedding.clone().to(token_embeds.device, dtype=token_embeds.dtype) 
        
        
        print(f'load embedding: {concept_name}')

        if tokenizer_idx==1:
            new_concept_cfg.update({
                concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }
            })
        elif tokenizer_idx==2:
            new_concept_cfg.update({
                concept_name: {
                    'concept_token_ids_2': new_token_ids,
                    'concept_token_names': new_token_names
                }
            })            

    return pipe, new_concept_cfg

def merge_lora_into_weight(original_state_dict, lora_state_dict, model_type, alpha):
    def get_lora_down_name(original_layer_name):
        if model_type == 'text_encoder' or model_type == 'text_encoder_2':
            lora_down_name = original_layer_name.replace('q_proj.weight', 'q_proj.lora_down.weight') \
                .replace('k_proj.weight', 'k_proj.lora_down.weight') \
                .replace('v_proj.weight', 'v_proj.lora_down.weight') \
                .replace('out_proj.weight', 'out_proj.lora_down.weight') \
                .replace('fc1.weight', 'fc1.lora_down.weight') \
                .replace('fc2.weight', 'fc2.lora_down.weight') \
                .replace('q.weight', 'q.lora_down.weight') \
                .replace('k.weight', 'k.lora_down.weight') \
                .replace('v.weight', 'v.lora_down.weight') \
                .replace('o.weight', 'o.lora_down.weight')
        else:
            lora_down_name = k.replace('to_q.weight', 'to_q.lora_down.weight') \
                .replace('to_k.weight', 'to_k.lora_down.weight') \
                .replace('to_v.weight', 'to_v.lora_down.weight') \
                .replace('to_out.0.weight', 'to_out.0.lora_down.weight') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj.lora_down.weight') \
                .replace('ff.net.2.weight', 'ff.net.2.lora_down.weight') \
                .replace('proj_out.weight', 'proj_out.lora_down.weight') \
                .replace('proj_in.weight', 'proj_in.lora_down.weight')

        return lora_down_name

    assert model_type in ['unet', 'text_encoder', 'text_encoder_2']
    new_state_dict = copy.deepcopy(original_state_dict)

    load_cnt = 0
    for k in new_state_dict.keys():
        lora_down_name = get_lora_down_name(k)
        lora_up_name = lora_down_name.replace('lora_down', 'lora_up')

        if lora_up_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            lora_down_params = lora_state_dict[lora_down_name].to(original_params.device)
            lora_up_params = lora_state_dict[lora_up_name].to(original_params.device)
            if len(original_params.shape) == 4:
                lora_param = lora_up_params.squeeze() @ lora_down_params.squeeze()
                lora_param = lora_param.unsqueeze(-1).unsqueeze(-1)
            else:
                lora_param = lora_up_params @ lora_down_params
            merge_params = original_params + alpha * lora_param
            new_state_dict[k] = merge_params

    print(f'load {load_cnt} LoRAs of {model_type}')
    return new_state_dict


def convert_edlora(pipe, state_dict, enable_edlora, alpha=0.6, n_layers=70):

    state_dict = state_dict['params'] if 'params' in state_dict.keys() else state_dict

    # step 1: load embedding
    if 'new_concept_embedding' in state_dict and len(state_dict['new_concept_embedding']) != 0:
        pipe, new_concept_cfg = load_new_concept(pipe, state_dict['new_concept_embedding'], enable_edlora, n_layers=n_layers)        
        
    if 'new_concept_embedding_2' in state_dict and len(state_dict['new_concept_embedding_2']) != 0:
        pipe, new_concept_cfg_2 = load_new_concept(pipe, state_dict['new_concept_embedding_2'], enable_edlora, tokenizer_idx=2, n_layers=n_layers)      
        for concept in new_concept_cfg.keys():
            new_concept_cfg[concept].update(new_concept_cfg_2[concept])  

    # step 2: merge lora weight to unet
    if 'unet' in state_dict:
        unet_lora_state_dict = state_dict['unet']
        pretrained_unet_state_dict = pipe.unet.state_dict()
        updated_unet_state_dict = merge_lora_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet', alpha=alpha)
        pipe.unet.load_state_dict(updated_unet_state_dict)
        del pretrained_unet_state_dict
        del unet_lora_state_dict
        torch.cuda.empty_cache()
        gc.collect()        

    # step 3: merge lora weight to text_encoder
    if 'text_encoder' in state_dict:
        text_encoder_lora_state_dict = state_dict['text_encoder']
        pretrained_text_encoder_state_dict = pipe.text_encoder.state_dict()
        updated_text_encoder_state_dict = merge_lora_into_weight(pretrained_text_encoder_state_dict, text_encoder_lora_state_dict, model_type='text_encoder', alpha=alpha)
        pipe.text_encoder.load_state_dict(updated_text_encoder_state_dict)
        del pretrained_text_encoder_state_dict
        del text_encoder_lora_state_dict
        torch.cuda.empty_cache()
        gc.collect()        
    
    if "text_encoder_2" in state_dict:
        text_encoder_2_lora_state_dict = state_dict['text_encoder_2']
        pretrained_text_encoder_2_state_dict = pipe.text_encoder_2.state_dict()
        updated_text_encoder_2_state_dict = merge_lora_into_weight(pretrained_text_encoder_2_state_dict, text_encoder_2_lora_state_dict, model_type='text_encoder_2', alpha=alpha)
        pipe.text_encoder_2.load_state_dict(updated_text_encoder_2_state_dict)    
        del pretrained_text_encoder_2_state_dict
        del text_encoder_2_lora_state_dict
        torch.cuda.empty_cache()
        gc.collect()

    # print('text_encoder_weight:')
    # print(updated_text_encoder_state_dict)    
    
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()

    return pipe, new_concept_cfg
