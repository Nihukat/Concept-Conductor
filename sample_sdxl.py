import sys
sys.path.append('./')

import argparse
from omegaconf import OmegaConf, AnyNode
import inspect


import math
import numpy as np
import torch

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

from diffusers.utils import logging

from concept_conductor.pipelines.pipeline_concept_conductor_sdxl import SDXLConceptConductorPipeline
from concept_conductor.models.earlybreak_unet import EarlyBreakUnet
from concept_conductor.utils import truncate_text

device = torch.device("cuda")

def dummy(images, **kwargs):
    return images, [False] * len(images)

def sample(args):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    variant = args.variant if args.variant else None
    pipeline =  SDXLConceptConductorPipeline.from_pretrained(args.sd_ckpt, variant=variant, revision="fp16", torch_dtype=torch.float16).to(device)
    pipeline.unet = EarlyBreakUnet.from_pretrained(args.sd_ckpt, variant=variant, revision="fp16", torch_dtype=torch.float16, subfolder='unet').to(device)

    pipeline.safety_checker = dummy
    if args.verbose:
        logging.enable_progress_bar()
    else:
        logging.disable_progress_bar()

    (Path(args.outroot)/Path(args.image_outdir)).mkdir(parents=True, exist_ok=True)
    sample_dir = str(Path(args.outroot) / Path(args.image_outdir) / Path(f"samples"))
    Path(sample_dir).mkdir(exist_ok=True)
    base_count = len(list(Path(sample_dir).iterdir()))
    
    all_images = []
    progress_bar = tqdm(total=args.n_batches * args.batch_size)
    
    word_list = []
    ref_token_ids = []
    for word_idx, word in enumerate(args.ref_prompt.split(' '), start=1):
        word_list.append(word.strip('`'))
        if '`' in word:
            ref_token_ids.append(word_idx)
    ref_prompt = ' '.join(word_list)
            
    all_token_ids = []
    prompts = []
    for prompt in [args.base_prompt] + args.custom_prompts:
        word_list = []
        token_ids = []
        for word_idx, word in enumerate(prompt.split(' '), start=1):
            word_list.append(word.strip('`'))
            if '`' in word:
                token_ids.append(word_idx)
        prompt = ' '.join(word_list)
        prompts.append(prompt)
        all_token_ids.append(token_ids)
    
    base_prompt = prompts[0]
    custom_prompts = prompts[1:]
    
    
    ref_image = Image.open(args.ref_image_path).convert("RGB")    
    
    ref_masks = []
    for ref_mask_path in args.ref_mask_paths:
        ref_mask = Image.open(ref_mask_path).convert("L").resize((args.width, args.height))
        ref_masks.append(ref_mask)        
    
    mask_center_points = []
    if args.init_mask_from_points and args.mask_center_points:
        for x, y in args.mask_center_points:
            new_x = x  * args.width // ref_image.width
            new_y = y  * args.height // ref_image.height 
            mask_center_points.append((new_x, new_y))
        
    ref_image = ref_image.resize((args.width, args.height))
    
    if not(args.init_mask_from_points) and args.init_image_path and args.init_mask_path:
        init_image = Image.open(args.init_image_path).convert("RGB").resize((args.width, args.height))
        init_mask = Image.open(args.init_mask_path).convert("L").resize((args.width, args.height))
    else:
        init_image = None
        init_mask = None
    
    edloras = None
    if args.load_edlora:
        edloras = []
        for edlora_path in args.edlora_paths:
            state_dict = torch.load(edlora_path)
            edlora = state_dict['params'] if 'params' in state_dict.keys() else state_dict
            edloras.append(edlora)



    seed = args.start_seed
    for batch_i in range(args.n_batches):
        prompts = []
        for j in range(args.batch_size):
            prompts.append(base_prompt)
        generator = [torch.Generator(device=device).manual_seed(i) for i in range(seed, seed + args.batch_size)]    
        sig = inspect.signature(pipeline.__call__)
        pipeline_params = sig.parameters.keys()
        args_dict = vars(args)['_content']
        excluded_params = ['prompt', 'ref_prompt', 'custom_prompts', 'mask_center_points', 'negative_prompt']
        
        filtered_args = {}
        for k, v in args_dict.items():
            if (k in pipeline_params) and not(k in excluded_params):
                if isinstance(v, AnyNode):
                    v = v._value()
                filtered_args[k] = v
         
        images = pipeline(prompts, generator=generator, negative_prompt="",
                            custom_prompts=custom_prompts, ref_prompt=ref_prompt,
                            all_token_ids=all_token_ids, ref_token_ids=ref_token_ids,
                            ref_image=ref_image, ref_masks=ref_masks, 
                            mask_center_points=mask_center_points, init_image=init_image, init_mask=init_mask,
                            edloras=edloras, **filtered_args).images
            
        for j, image in enumerate(images):
            truncated_text = truncate_text(args.base_prompt, 200)
            file_name = f'{base_count:05}_{truncated_text}_{(seed+j):02}.png'
            image.save(str(Path(sample_dir)/Path(file_name)))
            base_count += 1
            progress_bar.update(1)             
        
        all_images += images
        seed += args.batch_size
    progress_bar.close()
    file_name = f'{base_prompt.replace("/"," ")[:242]}_seed{args.start_seed}-{args.start_seed+args.n_batches*args.batch_size-1}.png'
    
    n_images = len(all_images)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)
    image_groups = []
    for i in range(n_rows):
        image_group = [np.array(x) for x in all_images[i * n_cols: (i + 1) * n_cols]]
        if len(image_group) < n_cols:
            image_group += [np.zeros_like(image_group[-1])] * (n_cols - len(image_group))
        image_groups.append(np.hstack(image_group))
    grid = np.vstack(image_groups)
    grid = Image.fromarray(grid)  
    grid.save(str(Path(args.outroot) / Path(args.image_outdir) / Path(file_name)))
    print(f"Your samples are ready and waiting for you here: \n{args.outroot}/{args.image_outdir} \n\nEnjoy.")

def load_config():
    default_config = OmegaConf.load('configs/sample_sdxl_config.yaml')
    
    parser = argparse.ArgumentParser('', add_help=False)
    
    parser.add_argument('--config_file', type=str, default='')
    
    parser.add_argument('--ref_prompt', type=str, default=default_config.inputs.ref_prompt)
    parser.add_argument('--base_prompt', type=str, default=default_config.inputs.base_prompt)
    parser.add_argument('--negative_prompt', type=str, default=default_config.inputs.negative_prompt)
    parser.add_argument('--custom_prompts', nargs='+', default=default_config.inputs.custom_prompts)
    parser.add_argument('--ref_image_path', type=str, default=default_config.inputs.ref_image_path)
    parser.add_argument('--ref_mask_paths', nargs='+', default=default_config.inputs.ref_mask_paths)
    parser.add_argument('--init_mask_from_points', action='store_true', default=default_config.inputs.init_mask_from_points)
    parser.add_argument('--mask_center_points', nargs='+', default=default_config.inputs.mask_center_points)
    parser.add_argument('--init_image_path', type=str, default=default_config.inputs.init_image_path)
    parser.add_argument('--init_mask_path', nargs='+', default=default_config.inputs.init_mask_path)    
    
    parser.add_argument('--edlora_paths', nargs='+', default=default_config.inputs.edlora_paths)
    parser.add_argument('--load_edlora', action='store_true', default=default_config.inputs.load_edlora)
    parser.add_argument('--lora_alpha', type=float, default=default_config.inputs.lora_alpha)
    
    parser.add_argument('--outroot', type=str, default=default_config.outputs.outroot)
    parser.add_argument('--image_outdir', type=str, default=default_config.outputs.image_outdir)
    parser.add_argument('--latents_outdir', type=str, default=default_config.outputs.latents_outdir)
    parser.add_argument('--self_attn_outdir', type=str, default=default_config.outputs.self_attn_outdir)    
    parser.add_argument('--cross_attn_outdir', type=str, default=default_config.outputs.cross_attn_outdir)
    parser.add_argument('--feature_mask_outdir', type=str, default=default_config.outputs.feature_mask_outdir)
    
    parser.add_argument('--sd_ckpt', type=str, default=default_config.base_model.sd_ckpt)
    parser.add_argument('--variant', type=str, default=default_config.base_model.variant)
    
    parser.add_argument('--height', type=int, default=default_config.sd_t2i.height)
    parser.add_argument('--width', type=int, default=default_config.sd_t2i.width)
    parser.add_argument('--guidance_scale', type=float, default=default_config.sd_t2i.guidance_scale)
    parser.add_argument('--num_inference_steps', type=int, default=default_config.sd_t2i.num_inference_steps)
    parser.add_argument('--start_seed', type=int, default=default_config.sd_t2i.start_seed)
    parser.add_argument('--batch_size', type=int, default=default_config.sd_t2i.batch_size)
    parser.add_argument('--n_batches', type=int, default=default_config.sd_t2i.n_batches)
    parser.add_argument('--verbose', action='store_true', default=default_config.sd_t2i.verbose)
    
    parser.add_argument('--attn_guidance_end', type=int, default=default_config.attention_operations.attn_guidance_end)
    parser.add_argument('--attn_guidance_interval', type=int, default=default_config.attention_operations.attn_guidance_interval)    
    parser.add_argument('--attn_guidance_weight', type=float, default=default_config.attention_operations.attn_guidance_weight)
    parser.add_argument('--custom_attn_guidance_factor', type=float, default=default_config.attention_operations.custom_attn_guidance_factor)    

    parser.add_argument('--processor_filter_guidance', type=str, default=default_config.attention_operations.processor_filter_guidance)
    parser.add_argument('--params_guidance', nargs='+', default=default_config.attention_operations.params_guidance)
    parser.add_argument('--processor_filter_mask', type=str, default=default_config.attention_operations.processor_filter_mask)
    parser.add_argument('--params_mask', nargs='+', default=default_config.attention_operations.params_mask)    
    parser.add_argument('--processor_filter_merge', type=str, default=default_config.attention_operations.processor_filter_merge)
    parser.add_argument('--params_merge', nargs='+', default=default_config.attention_operations.params_merge)    
    parser.add_argument('--processor_filter_view_sa', type=str, default=default_config.attention_operations.processor_filter_view_sa)
    parser.add_argument('--params_view_sa', nargs='+', default=default_config.attention_operations.params_view_sa)  
    parser.add_argument('--processor_filter_view_ca', type=str, default=default_config.attention_operations.processor_filter_view_ca)
    parser.add_argument('--params_view_ca', nargs='+', default=default_config.attention_operations.params_view_ca)      
    
    parser.add_argument('--mask_refinement_start', type=int, default=default_config.attention_operations.mask_refinement_start)
    parser.add_argument('--mask_refinement_end', type=int, default=default_config.attention_operations.mask_refinement_end)   
    parser.add_argument('--mask_update_interval', type=int, default=default_config.attention_operations.mask_update_interval)
    parser.add_argument('--mask_overlap_threshold', type=float, default=default_config.attention_operations.mask_overlap_threshold)
    parser.add_argument('--num_kmeans_init', type=int, default=default_config.attention_operations.num_kmeans_init)
    
    parser.add_argument('--rect_mask', action='store_true', default=default_config.attention_operations.rect_mask)
    parser.add_argument('--use_loss_mask', action='store_true', default=default_config.attention_operations.use_loss_mask)
    parser.add_argument('--visualization', action='store_true', default=default_config.attention_operations.visualization)
     
    
    args = parser.parse_args()
    config = OmegaConf.create(vars(args))
    
    
    if args.config_file:
        new_config = OmegaConf.load(args.config_file)
        for param_type in new_config.keys():
            for param_name in new_config[param_type].keys():
                config[param_name] = new_config[param_type][param_name]
            
        

    return config


if __name__ == "__main__":
    args = load_config()
    sample(args)
