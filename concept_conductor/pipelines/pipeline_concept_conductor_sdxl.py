# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is based on an Apache 2.0 licensed project: diffusers
# Original code: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import gc

from diffusers import StableDiffusionXLPipeline, DDIMInverseScheduler
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg, retrieve_timesteps

from concept_conductor.pipelines.attention_processor import AttnProcessor, AttentionController
from concept_conductor.utils import *


import torch.nn.functional as F
from pathlib import Path
import copy
import re
import time
import PIL
import torchvision.transforms as transforms

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from collections import defaultdict

torch.autograd.set_detect_anomaly(True)

class SDXLConceptConductorPipeline(StableDiffusionXLPipeline): 

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,        
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,        
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],        
        
        
        custom_prompts: List[str] = None,
        ref_prompt: str = None,
        all_token_ids: List[int] = None,
        ref_token_ids: List[int] = None,
        ref_image: PIL.Image.Image = None,
        ref_masks: List[PIL.Image.Image] = None,
        mask_center_points: List[List[int]] = None,
        init_image: PIL.Image.Image = None,
        init_mask:PIL.Image.Image = None,
        edloras: List = None,
        lora_alpha: int = 0.7,
        
        outroot: str = "outputs",
        latents_outdir: str = "inverted_latents",
        self_attn_outdir: str = "self_attn",
        cross_attn_outdir: str = "cross_attn",
        feature_mask_outdir: str = "feature_mask",
        
        attn_guidance_end: int = 60,
        attn_guidance_interval: int = 1,
        attn_guidance_weight: int = 10,
        custom_attn_guidance_factor: float = 1.0,
        
        processor_filter_guidance: str = '.*up_blocks\.1\.attentions\.0.*attn1.*',
        params_guidance: str = ["key"],
        processor_filter_mask: str = '.*up_blocks\.2\.attentions\.2.*attn1.*',
        params_mask: str = ['attention_probs'],
        processor_filter_merge: str = '.*up_blocks.*',
        params_merge: str = ["feature_output"],
        processor_filter_view_sa: str = '.*up_blocks\.2\.attentions\.2.*attn1.*',
        params_view_sa: str = ["attention_probs"],      
        processor_filter_view_ca: str = '.*up_blocks\.2\.attentions\.1.*attn2.*',
        params_view_ca: str = ["attention_probs"],    
        
        mask_refinement_start: int = 50,
        mask_refinement_end: int = 80,
        mask_update_interval: int = 5,
        mask_overlap_threshold: float = 0.7,
        num_kmeans_init: int = 50,
        rect_mask: bool = False,

        use_loss_mask: bool = False,
        visualization: bool = False,        

        **kwargs,
    ):
        """"""
        
        
        
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
                
            custom_prompts (List[str], optional): A list of all custom prompts. 
            ref_prompt (str, optional): Prompt corresponding to the reference image. 
            all_token_ids (List[int], optional): Ids of the tokens needed to be visualized when visualizing cross-attention in all generative branches.
            ref_token_ids (List[int], optional): Ids of the tokens needed to be visualized when visualizing cross-attention in the reference branch.
            ref_image (PIL.Image.Image, optional): A reference image for layout guidance.
            ref_masks (List[PIL.Image.Image], optional): Masks that specify the dense areas of the target subjects in the reference image.
            mask_center_points (List[List[int]], optional): Points that specify the locations of the target subjects in the reference image.
            init_image (PIL.Image.Image, optional): An image to be edited.
            init_mask (PIL.Image.Image, optional): A mask that specifies the area to be edited.
            edloras (List, optional): _description_. A list containing all the ED-LoRAs needed for multi-concept customization.
            lora_alpha (int, optional): A weight in the lora merge that controls the degree of influence on the original model.
            outroot (str, optional): Root directory of the output files.
            latents_outdir (str, optional): Directory for outputting inverted latents.
            self_attn_outdir (str, optional): Directory for outputting self-attention.
            cross_attn_outdir (str, optional): Directory for outputting cross-attention.
            feature_mask_outdir (str, optional): Directory for outputting feature masks.
            attn_guidance_end (int, optional): The last Step of Attentional Guidance.
            attn_guidance_interval (int, optional): Perform Attentional Guidance every how many steps.
            attn_guidance_weight (int, optional): _description_. Defaults to 10.
            custom_attn_guidance_factor (float, optional): _description_. Defaults to 1.0.
            processor_filter_guidance (str, optional): A regular expression used to filter the processors for layout guidance.
            params_guidance (str, optional): Self-attention parameters used for layout guidance.
            processor_filter_mask (str, optional): A regular expression used to filter the processors for self-attention clustering.
            params_mask (str, optional): Self-attention parameter for used to cluster to produce shape-aware masks.
            processor_filter_merge (str, optional): A regular expression used to filter the processors for feature fusion.
            params_merge (str, optional): Attention parameters to be merged for feature fusion.
            processor_filter_view_sa (str, optional): A regular expression used to filter the processors for self-attention visualization.
            params_view_sa (str, optional): Self-attention parameters to be visualized.
            processor_filter_view_ca (str, optional): A regular expression used to filter the processors for cross-attention visualization.
            params_view_ca (str, optional): Cross-attention parameters to be visualized.
            mask_refinement_start (int, optional): The first step of Mask Refinement.
            mask_refinement_end (int, optional): The last step of Mask Refinement.
            mask_update_interval (int, optional): Update the masks every how many steps.
            mask_overlap_threshold (float, optional): The mask will only be updated if the area of the new mask and the area of the old mask are greater than this threshold.
            rect_mask (bool, optional): Whether to use a rectangular mask to specify the region for feature fusion instead of a shape-aware mask based on self-attentive clustering.
            num_kmeans_init (int, optional): Number of times the k-means algorithm is run with different centroid seeds. 
            use_loss_mask (bool, optional): Whether to use a mask for layout-aligned loss thus controlling only the foreground.
            visualization (bool, optional): Whether to visualize some intermediate quantities, such as attention maps and feature masks.


        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """        

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        
        latents_outdir = str(Path(outroot)/ Path(latents_outdir))
        self_attn_outdir = str(Path(outroot)/ Path(self_attn_outdir))
        cross_attn_outdir = str(Path(outroot)/ Path(cross_attn_outdir))
        feature_mask_outdir = str(Path(outroot)/ Path(feature_mask_outdir))

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device        
        device = self.unet.device
        bs = batch_size
        n_cross_attn_layers = 70
        

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        

        self.text_encoder = self.text_encoder.to(device=device)
        self.text_encoder_2 = self.text_encoder_2.to(device=device)
        
        original_tokenizer = copy.deepcopy(self.tokenizer)
        original_text_encoder = copy.deepcopy(self.text_encoder)
        original_text_encoder_2 = copy.deepcopy(self.text_encoder_2)
        
        ref_prompt_embeds, _, ref_pooled_prompt_embeds, _ = self.encode_prompt(
            ref_prompt, ref_prompt,
            device,
            1,
            False,
            "", "",
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,            
        )
        ref_prompt_embeds = ref_prompt_embeds.detach()  # [1, 77, 2048]
        ref_pooled_prompt_embeds = ref_pooled_prompt_embeds.detach()    # [1, 1280]

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt, prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            [negative_prompt]*bs, [negative_prompt]*bs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        prompt_embeds = prompt_embeds.detach()  # [bs, 77, 2048]      
        negative_prompt_embeds = negative_prompt_embeds.detach()    # [bs, 77, 2048]
        pooled_prompt_embeds = pooled_prompt_embeds.detach()    # [bs, 1280]
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.detach()  # [bs, 1280]
        
        all_custom_prompt_embeds = []
        all_custom_pooled_prompt_embeds = []
        all_custom_negative_prompt_embeds = []
        all_custom_negative_pooled_prompt_embeds = []
        for rid, custom_prompt in enumerate(custom_prompts):
            
            
            if edloras:
                state_dict = edloras[rid]
                
                new_concept_embedding = state_dict['new_concept_embedding']
                new_concept_embedding_2 = state_dict['new_concept_embedding_2']
                new_prompts = [custom_prompt] * n_cross_attn_layers
                for idx, ((concept_name, concept_embedding), (_, concept_embedding_2)) \
                in enumerate(zip(new_concept_embedding.items(), new_concept_embedding_2.items())):
                             
                    new_token_names = [f'<new{idx * n_cross_attn_layers + layer_id}>' for layer_id in range(n_cross_attn_layers)]
                    
                    for concept_embed, tokenizer, text_encoder in zip([concept_embedding, concept_embedding_2],
                                                                          [self.tokenizer, self.tokenizer_2],
                                                                          [self.text_encoder, self.text_encoder_2]):
                        tokenizer.add_tokens(new_token_names)
                        new_token_ids = [tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]
                        text_encoder.resize_token_embeddings(len(tokenizer))
                        token_embeds = text_encoder.get_input_embeddings().weight.data
                        token_embeds[new_token_ids] = concept_embed.clone().to(device=device, dtype=token_embeds.dtype)
                        
                    
                    for layer_id in range(n_cross_attn_layers):
                        new_prompts[layer_id] = new_prompts[layer_id].replace(concept_name, new_token_names[layer_id])
                    
                text_encoder_lora_state_dict = state_dict['text_encoder']
                pretrained_text_encoder_state_dict = self.text_encoder.state_dict()
                updated_text_encoder_state_dict = merge_lora_into_weight(pretrained_text_encoder_state_dict, text_encoder_lora_state_dict, model_type='text_encoder', alpha=lora_alpha)
                self.text_encoder.load_state_dict(updated_text_encoder_state_dict)    
                del text_encoder_lora_state_dict
                gc.collect()
                torch.cuda.empty_cache()  
                
                text_encoder_2_lora_state_dict = state_dict['text_encoder_2']
                pretrained_text_encoder_2_state_dict = self.text_encoder_2.state_dict()
                updated_text_encoder_2_state_dict = merge_lora_into_weight(pretrained_text_encoder_2_state_dict, text_encoder_2_lora_state_dict, model_type='text_encoder_2', alpha=lora_alpha)
                self.text_encoder_2.load_state_dict(updated_text_encoder_2_state_dict)    
                del text_encoder_2_lora_state_dict
                gc.collect()
                torch.cuda.empty_cache()                  
                
                custom_prompt_embeds, _, custom_pooled_prompt_embeds, _ = self.encode_prompt(
                    new_prompts,
                    new_prompts,
                    device,
                    num_images_per_prompt,
                    False
                )
                
                custom_negative_prompt_embeds, _, custom_negative_pooled_prompt_embeds, _ = self.encode_prompt(
                    negative_prompt,
                    negative_prompt,
                    device,
                    num_images_per_prompt,
                    False
                )
                
                custom_prompt_embeds = custom_prompt_embeds.unsqueeze(dim=0).repeat(bs,1,1,1) # [bs, n_layers, 77, 2048]
                custom_negative_prompt_embeds = custom_negative_prompt_embeds.unsqueeze(dim=0).repeat(bs,n_cross_attn_layers,1,1)  # [bs, n_layers, 77, 2048]
                
                custom_pooled_prompt_embeds = custom_pooled_prompt_embeds[0].unsqueeze(dim=0).repeat(bs,1)  # [bs, 1280]
                custom_negative_pooled_prompt_embeds = custom_negative_pooled_prompt_embeds.repeat(bs,1)    # [bs, 1280]

                self.tokenizer = copy.deepcopy(original_tokenizer)
                self.text_encoder = copy.deepcopy(original_text_encoder)  
                self.text_encoder_2 = copy.deepcopy(original_text_encoder_2)                                               

            else:
                custom_prompt_embeds, custom_negative_prompt_embeds, custom_pooled_prompt_embeds, custom_negative_pooled_prompt_embeds = self.encode_prompt(
                    custom_prompt, custom_prompt,
                    device,
                    num_images_per_prompt, 
                    self.do_classifier_free_guidance,
                    negative_prompt, negative_prompt,
                )
                custom_prompt_embeds = custom_prompt_embeds.repeat(bs,1,1)  # [bs, 77, 2048]
                custom_negative_prompt_embeds = custom_negative_prompt_embeds.repeat(bs,1,1)    # [bs, 77, 2048]
                custom_pooled_prompt_embeds = custom_pooled_prompt_embeds.repeat(bs, 1) # [bs, 1280]
                custom_negative_pooled_prompt_embeds = custom_negative_pooled_prompt_embeds.repeat(bs, 1)   # [bs, 1280]
                

            custom_prompt_embeds = custom_prompt_embeds.detach()
            custom_negative_prompt_embeds = custom_negative_prompt_embeds.detach()
            custom_pooled_prompt_embeds = custom_pooled_prompt_embeds.detach()
            custom_negative_pooled_prompt_embeds = custom_negative_pooled_prompt_embeds.detach()
            all_custom_prompt_embeds.append(custom_prompt_embeds)
            all_custom_negative_prompt_embeds.append(custom_negative_prompt_embeds)
            all_custom_pooled_prompt_embeds.append(custom_pooled_prompt_embeds)
            all_custom_negative_pooled_prompt_embeds.append(custom_negative_pooled_prompt_embeds)

        del original_tokenizer
        del original_text_encoder   
        del original_text_encoder_2
        gc.collect()
        torch.cuda.empty_cache()             


        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )        
            
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )        
        
        noise = latents / self.scheduler.init_noise_sigma
        
        ref_latents_dict = self.invert(image=ref_image, prompt=ref_prompt, 
                                       prompt_embeds=ref_prompt_embeds, pooled_prompt_embeds=ref_pooled_prompt_embeds,
                                       latents_outdir=latents_outdir)
        
         
        if init_image and init_mask:
            init_image = self.image_processor.preprocess(init_image).to(dtype=self.vae.dtype, device=self.vae.device)   # [1, 3, h, w]
            init_latents = self.vae.encode(init_image).latent_dist.mean
            init_latents = self.vae.config.scaling_factor * init_latents    # [1, 4, h//8, w//8]
            init_mask = transforms.Resize((height // 8, width // 8))(init_mask)
            init_mask = transforms.PILToTensor()(init_mask)
            init_mask = init_mask > 127 # [1, h//8, w//8]
            init_mask = init_mask.to(dtype=latents.dtype, device=device)        
            init_mask = init_mask.unsqueeze(dim=1)  # [1, 1, h//8, w//8]

        self.text_encoder = self.text_encoder.to('cpu')
        self.text_encoder_2 = self.text_encoder_2.to('cpu')
        self.vae = self.vae.to('cpu')        
        
        if ref_masks:
            feature_masks = []
            for mask_id, ref_mask in enumerate(ref_masks):
                ref_mask = transforms.Resize((height // 8, width // 8))(ref_mask)
                ref_mask = transforms.PILToTensor()(ref_mask)
                feature_mask = ref_mask > 127
                feature_mask = feature_mask.to(dtype=latents.dtype, device=device)

                feature_masks.append(feature_mask)  # [1, 128, 128]        
        else:
            feature_masks = None          
            
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids        
        

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
            
            
        # Initialize an attention controller.
        def ensure_list(param):
            return [param] if isinstance(param, str) else param

        params_guidance = ensure_list(params_guidance)
        params_mask = ensure_list(params_mask)
        params_merge = ensure_list(params_merge)
        params_view_sa = ensure_list(params_view_sa)
        params_view_ca = ensure_list(params_view_ca)
        
        ops_dict = defaultdict(lambda : defaultdict(list))
        processors_guidance = []
        processors_mask = []
        processors_merge = []
        processors_view_sa = []
        processors_view_ca = []
        for processor_name in self.unet.attn_processors.keys():
            if re.match(processor_filter_guidance, processor_name):
                processors_guidance.append(processor_name)
                for param in params_guidance:
                    ops_dict[processor_name][param].append('guidance')
            if re.match(processor_filter_mask, processor_name):
                processors_mask.append(processor_name)
                for param in params_mask:
                    ops_dict[processor_name][param].append('mask')                    
            if re.match(processor_filter_merge, processor_name):
                processors_merge.append(processor_name)
                for param in params_merge:
                    ops_dict[processor_name][param].append('merge')
            if (visualization) and (re.match(processor_filter_view_sa, processor_name)):
                processors_view_sa.append(processor_name)
                for param in params_view_sa:
                    ops_dict[processor_name][param].append('view')
            if re.match(processor_filter_view_ca, processor_name):
                processors_view_ca.append(processor_name)
                for param in params_view_ca:
                    ops_dict[processor_name][param].append('view')     
                    
        max_guidance_pos = [-1, -1, -1]
        for processor_name in processors_guidance:
            temp_strs = processor_name.split('.')
            max_guidance_pos[0] = max(max_guidance_pos[0], int(temp_strs[1]))
            max_guidance_pos[1] = max(max_guidance_pos[1], int(temp_strs[3]))
            max_guidance_pos[2] = max(max_guidance_pos[2], int(temp_strs[5]))                          
        
        attention_controller = AttentionController(ops_dict=ops_dict, w_min=width//32, h_min=height//32,
                                         mask_overlap_threshold=mask_overlap_threshold, 
                                         num_kmeans_init=num_kmeans_init,
                                         all_token_ids=all_token_ids, ref_token_ids=ref_token_ids, 
                                         max_guidance_pos = max_guidance_pos,
                                         device=self.unet.device, dtype=self.unet.dtype, rect_mask=rect_mask) 


        # Get the index of each processor in ED-LoRA.

        all_processors = []
        def get_processors(unet, count):
            for name, layer in unet.named_children():
                if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                    all_processors.append(layer.processor)
                    count += 1
                else:
                    count = get_processors(layer, count)
            return count
        
        count = get_processors(self.unet.down_blocks, 0)
        count = get_processors(self.unet.mid_block, count)
        count = get_processors(self.unet.up_blocks, count)

        processor_indices = []
        for  processor_name, processor in self.unet.attn_processors.items():
            if "attn2" in processor_name:
                if processor in all_processors:
                    idx = all_processors.index(processor)
                    processor_indices.append(idx)

        # Prepare attention processors.
        base_attn_processors  = dict()
        cross_attn_count = 0
        for  processor_name in self.unet.attn_processors.keys():
            if "attn2" in processor_name:
                cross_attn_idx = processor_indices[cross_attn_count]
                cross_attn_count += 1
            else:
                cross_attn_idx = None
            base_attn_processors[processor_name] = AttnProcessor(attention_controller, processor_name, 0, 
                                                                   cross_attn_idx=cross_attn_idx, use_xformers=True, head_size=10)
            
        original_attn_processors = copy.deepcopy(self.unet.attn_processors)
            
        self.unet.set_attn_processor(copy.copy(base_attn_processors))  

        
        all_custom_attn_processors = []
        for rid in range(len(custom_prompts)):
            custom_attn_processors = dict()
            cross_attn_count = 0
            for  processor_name in self.unet.attn_processors.keys():
                if "attn2" in processor_name:
                    cross_attn_idx = processor_indices[cross_attn_count]
                    cross_attn_count += 1
                else:
                    cross_attn_idx = None                
                custom_attn_processors[processor_name] = AttnProcessor(attention_controller, processor_name, rid+1,
                                                                         cross_attn_idx = cross_attn_idx,
                                                                         use_xformers=True, head_size=10)
            all_custom_attn_processors.append(custom_attn_processors)  
        
        
        # If no predefined masks exist, masks are extracted from the reference graph based on the given points using self-attentive clustering.
        if (feature_masks is None) or (len(feature_masks) == 0):
            mid_t = timesteps[mask_refinement_end]
            ref_latents = ref_latents_dict[mid_t.tolist()].to(device)
            attention_controller.step = mask_refinement_end
            

            
            with torch.no_grad():
                attention_controller.batch_format = "ref"
                add_text_embeds = ref_pooled_prompt_embeds
                added_cond_kwargs['text_embeds'] = add_text_embeds        
                added_cond_kwargs["time_ids"] = add_time_ids.to(device).repeat(ref_pooled_prompt_embeds.shape[0], 1)              
                noise_pred = self.unet( 
                    ref_latents,
                    mid_t,
                    encoder_hidden_states=ref_prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]        

        if feature_masks or mask_center_points:
            # Initialize masks.
            attention_controller.init_feature_masks(feature_masks=feature_masks, points=mask_center_points, num_clusters=6, bs=batch_size)
        attention_controller.step = 0
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:

            for step, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                attention_controller.step = step

                
                
                requires_attn_guidance = (step <= attn_guidance_end) and (step % attn_guidance_interval == 0)
                
                # Attention-based layout guidance.
                if requires_attn_guidance:
                    attention_controller.mode = 'attn_guidance'
                    
                    ref_latents = ref_latents_dict[t.tolist()].to(device)
                    
                    
                    latents = latents.detach()
                    latents.requires_grad_(True)
                    # latent_model_input = torch.cat([latents, ref_latents])
                    latent_model_input = latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    attention_controller.batch_format = "cond"
                    attention_controller.requires_merge = False
                    for rid in range(len(custom_prompts)):
                        self.unet.set_attn_processor(copy.deepcopy(original_attn_processors))
                        custom_unet = copy.deepcopy(self.unet)
                        custom_unet.set_attn_processor(copy.copy(all_custom_attn_processors[rid]))
                        if edloras:
                            state_dict = edloras[rid]
                            unet_lora_state_dict = state_dict['unet']
                            pretrained_unet_state_dict = custom_unet.state_dict()
                            updated_unet_state_dict = merge_lora_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet', alpha=lora_alpha)
                            custom_unet.load_state_dict(updated_unet_state_dict)
                            
                            del unet_lora_state_dict
                            del state_dict
                            del updated_unet_state_dict
                            del pretrained_unet_state_dict         
                            
                            gc.collect()
                            torch.cuda.empty_cache()                   

                        add_text_embeds = all_custom_pooled_prompt_embeds[rid]  # [bs, 1280]
                        added_cond_kwargs['text_embeds'] = add_text_embeds
                        added_cond_kwargs["time_ids"] = add_time_ids.to(device).repeat(add_text_embeds.shape[0], 1)
                        custom_noise_pred = custom_unet(
                            latent_model_input[:bs],
                            t,
                            encoder_hidden_states=all_custom_prompt_embeds[rid],    # [2, 70, 77, 2048]
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False
                        )[0]
                        
                        del custom_noise_pred
                        del custom_unet
                        
                        
                        
                        gc.collect()
                        torch.cuda.empty_cache()
   
                    self.unet.set_attn_processor(copy.copy(base_attn_processors))  
                    
                    with torch.no_grad():
                        attention_controller.batch_format = "ref"
                        add_text_embeds = ref_pooled_prompt_embeds
                        added_cond_kwargs['text_embeds'] = add_text_embeds 
                        added_cond_kwargs["time_ids"] = add_time_ids.to(device).repeat(ref_pooled_prompt_embeds.shape[0], 1)                       
                        noise_pred = self.unet( 
                            ref_latents,
                            t,
                            encoder_hidden_states=ref_prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False
                        )[0]   
                        del noise_pred        
                        
                    gc.collect()
                    torch.cuda.empty_cache()                                                                 


                    attention_controller.batch_format = "cond"
                    add_text_embeds = pooled_prompt_embeds
                    added_cond_kwargs['text_embeds'] = add_text_embeds
                    added_cond_kwargs["time_ids"] = add_time_ids.to(device).repeat(pooled_prompt_embeds.shape[0], 1)                    
                    noise_pred = self.unet( 
                        latent_model_input[:bs],
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0] 
                    
                    del noise_pred      

                    gc.collect()
                    torch.cuda.empty_cache()                      
                                
                    layout_loss = self._compute_layout_loss(attention_controller,processors_guidance=processors_guidance, params_guidance=params_guidance, 
                                                                  custom_attn_guidance_factor=custom_attn_guidance_factor, use_loss_mask=use_loss_mask)
                    
                    # Update input latents with gradient descent.
                    gradient = torch.autograd.grad(layout_loss, latents, allow_unused=True)[0]  
                    
                    score = gradient[:bs] * attn_guidance_weight  
                    latents.requires_grad_(False)
                    latents -= score[:bs]

                    attention_controller.empty(keep_ref=True)
                    
                    del gradient
                    del score
                    del layout_loss
                    
                    
                    
                    gc.collect()
                    torch.cuda.empty_cache()     
                    
                    attention_controller.mode = 'generation'                 

                # Inject the Personalized concepts during the denoising process.
                attention_controller.requires_merge = True
                with torch.no_grad():  

                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).detach()                      
                   
                    attention_controller.batch_format = 'cond+uncond'
                    for rid in range(len(custom_prompts)):
                        self.unet.set_attn_processor(copy.deepcopy(original_attn_processors))
                        custom_unet = copy.deepcopy(self.unet)
                        custom_unet.set_attn_processor(copy.copy(all_custom_attn_processors[rid]))
                        if edloras:
                            state_dict = edloras[rid]
                            unet_lora_state_dict = state_dict['unet']
                            pretrained_unet_state_dict = custom_unet.state_dict()
                            updated_unet_state_dict = merge_lora_into_weight(pretrained_unet_state_dict, unet_lora_state_dict, model_type='unet', alpha=lora_alpha)
                            custom_unet.load_state_dict(updated_unet_state_dict)  
                            
                            del state_dict
                            del unet_lora_state_dict
                            del pretrained_unet_state_dict
                            del updated_unet_state_dict                                         
                            
                            
                        add_text_embeds = torch.cat([all_custom_pooled_prompt_embeds[rid], all_custom_negative_pooled_prompt_embeds[rid]])
                        added_cond_kwargs['text_embeds'] = add_text_embeds
                        added_cond_kwargs["time_ids"] = torch.cat([add_time_ids, negative_add_time_ids], dim=0).to(device).repeat(all_custom_pooled_prompt_embeds[rid].shape[0], 1)    
                        custom_noise_pred = custom_unet(  
                            latent_model_input,
                            t,
                            encoder_hidden_states=torch.cat([all_custom_prompt_embeds[rid], all_custom_negative_prompt_embeds[rid]]),
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                        
                        
                        del custom_noise_pred
                        del custom_unet
                        
                        gc.collect()
                        torch.cuda.empty_cache()
               
                    self.unet.set_attn_processor(copy.copy(base_attn_processors))


                    add_text_embeds = torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0)
                    added_cond_kwargs['text_embeds'] = add_text_embeds
                    added_cond_kwargs["time_ids"] = torch.cat([add_time_ids, negative_add_time_ids], dim=0).to(device).repeat(pooled_prompt_embeds.shape[0],1)
                    noise_pred = self.unet( 
                        latent_model_input,
                        t,
                        encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds]),
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]   
                    
                    if visualization and ((step % 10 == 0) or (step < 3)):
                        attention_controller.view_cross_attn(processors_view_ca, cross_attn_outdir)  
                        attention_controller.view_self_attn(processors_view_sa, self_attn_outdir, num_clusters=6) 
                        
                    if visualization and ((step % 5 == 0) or (step < 3)):    
                        attention_controller.view_feature_mask(feature_mask_outdir)
                        attention_controller.view_feature_mask(feature_mask_outdir.strip('/')+'_custom', prefix="custom_")
                        attention_controller.view_feature_mask(feature_mask_outdir.strip('/')+'_base', prefix="base_")
                    
                    # Mask refinement.
                    if  (step >= mask_refinement_start) and (step <= mask_refinement_end) and (step % mask_update_interval == 0) and (feature_masks or mask_center_points):
                        # print('\nrefinemnt\n')
                        attention_controller.refine_feature_masks()      
                    # else:
                    #     print(f'\nstep:{step} start:{args.mask_refinement_start} end:{args.mask_refinement_end} interval:{args.mask_update_interval}\n')                          

                attention_controller.empty()
                del latent_model_input
                gc.collect()
                torch.cuda.empty_cache()     
                
                             

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    del noise_pred_text
                    del noise_pred_uncond
              

                    # score = torch.cat([score, torch.zeros_like(score)])

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
           
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                    


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]  
                # latents = self.scheduler.step(noise_pred, t, latents, score=score, guidance_scale=attn_guidance_scale,
                #                               **extra_step_kwargs, return_dict=False)[0]
                
                
                if (init_image is not None) and (init_mask is not None):
                    init_latents_proper = init_latents
                    if step < len(timesteps) - 1:
                        noise_timestep = timesteps[step + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                    
                
                del noise_pred
                gc.collect()
                torch.cuda.empty_cache()                     


                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, step, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if step == len(timesteps) - 1 or ((step + 1) > num_warmup_steps and (step + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and step % callback_steps == 0:
                        step_idx = step // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                        
        del attention_controller
        gc.collect()
        torch.cuda.empty_cache() 
        
        if not output_type == "latent":
            self.vae = self.vae.to(device=device)
            with torch.no_grad():
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]    
        else:
            image = latents


        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image)

        return StableDiffusionXLPipelineOutput(images=image)
            

    # Calculate the self-attention loss between the reference branch and generated branches for layout alignment.
    def _compute_layout_loss(self, attention_controller, processors_guidance, params_guidance,  custom_attn_guidance_factor=1.0, use_loss_mask=False):
        
        loss_list = []
        for processor_name in processors_guidance:
            for param_name in params_guidance:
                ref_attn = attention_controller.extract('ref', processor_name, param_name).detach()
                
                factor = int(np.sqrt(ref_attn.shape[1] // (attention_controller.h_min*attention_controller.w_min)))
                
                if use_loss_mask:
                    prefix = ''
                    ref_mask = attention_controller.extract('ref', '', f'{prefix}feature_mask_{factor}')
                    foreground_mask = ref_mask.sum(dim=-1, keepdim=True)
                else:
                    foreground_mask = torch.ones_like(ref_attn)

                model_losses = [] 

                for model_idx in range(len(attention_controller)-1):
                    attn = attention_controller.extract(model_idx, processor_name, param_name)
                    
                    model_loss = (F.mse_loss(attn, ref_attn, reduction='none') * foreground_mask).sum(dim=0).mean()
                    model_losses.append(model_loss)
   
                    
                current_loss = model_losses[0] + torch.stack(model_losses[1:], dim=0).mean(dim=0) * custom_attn_guidance_factor  


                loss_list.append(current_loss) 
        layout_loss = torch.stack(loss_list, dim=0).mean(dim=0)
        return layout_loss                          
        
        
        
    @torch.no_grad()
    def invert(self,
               image: PIL.Image.Image = None, prompt: str = None, 
               prompt_embeds: Optional[torch.Tensor] = None, pooled_prompt_embeds: Optional[torch.Tensor] = None,
               num_inference_steps=999, latents_outdir="inverted_latents"):
        
        self.vae = self.vae.to(device=self.unet.device)        
        
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        
        hash_key = image_to_hash(image)
        image_info = load_image_info(latents_outdir, hash_key, prompt)
        
        if image_info is not None:
            latents_dict = image_info['latents_dict']
            return latents_dict
        else:
            latents_dict = self.ddim_invert(prompt=prompt, image=image, num_inference_steps=num_inference_steps, 
                                            prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)
            image_info = {
                'hash_key': hash_key,
                'prompt': prompt,
                'latents_dict': latents_dict,
            }
            save_image_info(image_info, latents_outdir)
            return latents_dict
        
    # Based on https://github.com/dvirsamuel/FPI/blob/main/StableDiffusionPipelineWithDDIMInversion.py
    @torch.no_grad()
    def ddim_invert(
            self,
            prompt: Optional[str] = None,
            image: PIL.Image.Image = None,
            num_inference_steps: int = 50,
            prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None
    ):

        device = self._execution_device

        image = self.image_processor.preprocess(image).to(dtype=self.vae.dtype, device=self.vae.device)
        latents = latents = self.vae.encode(image).latent_dist.mean
        latents = self.vae.config.scaling_factor * latents
        
        height = latents.shape[1]
        width = latents.shape[2]
        
        original_size = (height, width)
        target_size = (height, width)
        
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim        
        
        
        if (prompt_embeds) is None or (pooled_prompt_embeds is None):
            prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(
                prompt, prompt,
                device,
                1,
                False,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds
        )

        # 6. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        latents_dict = {}
        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step, t in enumerate(timesteps):
                timestep = t.detach().cpu().item()
                latent_model_input = self.inverse_scheduler.scale_model_input(latents, t)

                added_cond_kwargs = {}
                
                       

                add_text_embeds = pooled_prompt_embeds
                added_cond_kwargs['text_embeds'] = add_text_embeds
                
                
                add_time_ids = self._get_add_time_ids(
                    original_size,
                    (0,0),
                    target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
                added_cond_kwargs["time_ids"] = add_time_ids.to(device).repeat(pooled_prompt_embeds.shape[0], 1)      
                             
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

                latents_dict[timestep] = latents.detach().cpu()

                # call the callback, if provided
                if step == len(timesteps) - 1 or (
                        (step + 1) > num_warmup_steps and (step + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()

        return latents_dict     
            