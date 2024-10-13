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

import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate
from typing import Optional
import xformers
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import gc
from PIL import Image
from torch_kmeans import KMeans
from concept_conductor.utils import gen_n_colors

class AttnProcessor:
    def __init__(self, attention_controller, processor_name, branch_idx, cross_attn_idx=None, use_xformers=True, attention_op=None, head_size=8):
        super().__init__()
        self.attention_controller = attention_controller
        self.processor_name = processor_name
        self.branch_idx = branch_idx
        self.cross_attn_idx = cross_attn_idx
        self.use_xformers=use_xformers
        self.attention_op=attention_op
        self.head_size = head_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True
            if len(encoder_hidden_states.shape) == 4:
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attn_idx]            

        assert not attn.norm_cross

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)


        if self.attention_controller.batch_format == 'ref':
            bs = 0
        elif self.attention_controller.batch_format == 'cond+ref':
            bs = batch_size - 1
        elif self.attention_controller.batch_format == 'cond':
            bs = batch_size
        elif self.attention_controller.batch_format == 'cond+uncond':
            bs = batch_size // 2
        else:
            raise NotImplementedError

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
            
        if self.processor_name in self.attention_controller.ops_dict.keys():
            for param_name in self.attention_controller.ops_dict[self.processor_name].keys():
                param_ops = self.attention_controller.ops_dict[self.processor_name][param_name]
                if ('mask' in param_ops) or ('view' in param_ops):
                    if param_name == "feature_input":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, hidden_states[:bs].detach())
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, hidden_states[-1].unsqueeze(dim=0).detach())
                if ('guidance' in param_ops):
                    if param_name == "feature_input":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, hidden_states[:bs])
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, hidden_states[-1].unsqueeze(dim=0).detach())                            
                if (('merge' in param_ops) and (self.attention_controller.requires_merge) and (self.branch_idx > 0)):
                    if param_name == "feature_input":
                        self.attention_controller.store(self.branch_idx, self.processor_name, param_name, hidden_states.detach())                
                if (('merge' in param_ops) and (self.attention_controller.requires_merge) and (self.branch_idx == 0) and (batch_size % 2 == 0)):
                    if param_name == "feature_input":
                        hidden_states = self.merge(hidden_states, "feature_input")

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)      
        
        if self.processor_name in self.attention_controller.ops_dict.keys():
            for param_name in self.attention_controller.ops_dict[self.processor_name].keys():
                param_ops = self.attention_controller.ops_dict[self.processor_name][param_name]
                if ('mask' in param_ops) or ('view' in param_ops):
                    if param_name == "query":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, query[:bs].detach())
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, query[-1].unsqueeze(dim=0).detach())
                    if param_name == "key":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, key[:bs].detach())
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, key[-1].unsqueeze(dim=0).detach())
                    if param_name == "value":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, value[:bs].detach())
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, value[-1].unsqueeze(dim=0).detach())
                if ('guidance' in param_ops):
                    if param_name == "query":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, query[:bs])
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, query[-1].unsqueeze(dim=0).detach())
                    if param_name == "key":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, key[:bs])
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, key[-1].unsqueeze(dim=0).detach())
                    if param_name == "value":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, value[:bs])
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, param_name, value[-1].unsqueeze(dim=0).detach())                            
                if (('merge' in param_ops) and (self.attention_controller.requires_merge) and (self.branch_idx > 0)):
                    if param_name == "query":
                        self.attention_controller.store(self.branch_idx, self.processor_name, param_name, query.detach())
                    if param_name == "key":
                        self.attention_controller.store(self.branch_idx, self.processor_name, param_name, key.detach())
                    if param_name == "value":
                        self.attention_controller.store(self.branch_idx, self.processor_name, param_name, value.detach())                                            
                if (('merge' in param_ops) and (self.attention_controller.requires_merge) and (self.branch_idx == 0) and (batch_size % 2 == 0)):
                    if param_name == "query":
                        query = self.merge(query, "query")                         

        if self.use_xformers:
            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            hidden_states = hidden_states.to(query.dtype)
            attention_probs = None            
        else:
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        
        if self.processor_name in self.attention_controller.ops_dict.keys():
            for param_name in self.attention_controller.ops_dict[self.processor_name].keys():
                param_ops = self.attention_controller.ops_dict[self.processor_name][param_name]
                if ('mask' in param_ops) or ('view' in param_ops):
                    if param_name == "attention_probs":
                        if attention_probs is None:
                            with torch.no_grad():
                                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                        batch_size_, seq_len, dim = attention_probs.shape
                        head_size = self.head_size
                        attention_probs = attention_probs.reshape(batch_size_ // head_size, head_size, seq_len, dim).mean(dim=1)
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, "attention_probs", attention_probs[:bs].detach())
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, "attention_probs", attention_probs[-1].unsqueeze(dim=0).detach())
                    if param_name == "feature_output":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, hidden_states[:bs].detach())
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, "feature_output", hidden_states[-1].unsqueeze(dim=0).detach())
                if ('guidance' in param_ops):
                    if param_name == "attention_probs":
                        if attention_probs is None:
                            with torch.no_grad():
                                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                        batch_size_, seq_len, dim = attention_probs.shape
                        head_size = self.head_size
                        attention_probs = attention_probs.reshape(batch_size_ // head_size, head_size, seq_len, dim).mean(dim=1)
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, "attention_probs", attention_probs[:bs])
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, "attention_probs", attention_probs[-1].unsqueeze(dim=0).detach())
                    if param_name == "feature_output":
                        if bs > 0:
                            self.attention_controller.store(self.branch_idx, self.processor_name, param_name, hidden_states[:bs])
                        if 'ref' in self.attention_controller.batch_format:
                            self.attention_controller.store('ref', self.processor_name, "feature_output", hidden_states[-1].unsqueeze(dim=0).detach())                            
                if (('merge' in param_ops) and (self.attention_controller.requires_merge) and (self.branch_idx > 0)):
                    if param_name == "attention_probs":
                        if attention_probs is None:
                            with torch.no_grad():
                                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                        batch_size_, seq_len, dim = attention_probs.shape
                        head_size = self.head_size
                        attention_probs = attention_probs.reshape(batch_size_ // head_size, head_size, seq_len, dim).mean(dim=1)
                        self.attention_controller.store(self.branch_idx, self.processor_name, "attention_probs", attention_probs.detach())
                    if param_name == "feature_output":
                        self.attention_controller.store(self.branch_idx, self.processor_name, param_name, hidden_states.detach())
                if (('merge' in param_ops) and (self.attention_controller.requires_merge) and (self.branch_idx == 0) and (batch_size % 2 == 0)):
                    if param_name == "feature_output":
                        hidden_states = self.merge(hidden_states, "feature_output")
                        
                        
        if (self.attention_controller.mode=='attn_guidance'):
            if ('up_blocks' in self.processor_name):
                temp_strs = self.processor_name.split('.')
                max_pos = self.attention_controller.max_guidance_pos
                pos0 = int(temp_strs[1])
                pos1 = int(temp_strs[3])
                pos2 = int(temp_strs[5])
                if ((pos0 > max_pos[0]) \
                    or ((pos0 == max_pos[0]) and (pos1 > max_pos[1])) \
                    or ((pos0 == max_pos[0]) and (pos1 == max_pos[1]) and (pos2 >= max_pos[2]))):
                    raise Exception("Early Break: Attention Guidance")    # All the features needed for attention guidance have been stored

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def merge(self, base_feature, feature_name):
        
        for idx in range(1, len(self.attention_controller) - 1):
            custom_feature = self.attention_controller.extract(idx, self.processor_name, feature_name)
            
            factor = int(np.sqrt(custom_feature.shape[1] // (self.attention_controller.h_min*self.attention_controller.w_min)))
            
            
            custom_feature_mask = self.attention_controller.extract(idx, '', f'feature_mask_{factor}')
            if custom_feature.shape[0] == custom_feature_mask.shape[0] * 2:
                custom_feature_mask = torch.cat([custom_feature_mask, custom_feature_mask], dim=0)
            a = custom_feature_mask.max()
            base_feature = custom_feature * custom_feature_mask + base_feature * (1 - custom_feature_mask)   
        return base_feature      
    
    
class AttentionController(object):
    def __init__(self, ops_dict, w_min=8, h_min=8,
                 all_token_ids=None, ref_token_ids=None, 
                 max_guidance_pos=[-1, -1, -1],
                 mask_overlap_threshold=0.5, num_kmeans_init=100,
                 device="cuda", dtype=torch.float16, rect_mask=False):
        self.storage = dict()
        self.ops_dict = ops_dict
        
        self.w_min = w_min
        self.h_min = h_min
        
        
        self.all_token_ids = all_token_ids
        self.ref_token_ids = ref_token_ids
        self.max_guidance_pos = max_guidance_pos
        
        self.mask_overlap_threshold = mask_overlap_threshold
        self.num_kmeans_init = num_kmeans_init
        
        self.device = device
        self.dtype = dtype
        
        self.batch_format = 'cond+uncond'
        self.mode = 'generation'
        self.requires_merge = True
        
        self.rect_mask = rect_mask
        
    def __len__(self):
        return len(self.storage)
        
    def extract(self, branch_idx, processor_name, param_name):
        try:
            param = self.storage[branch_idx][processor_name][param_name]
        except:
            param = None
        return param           


    def empty(self, keep_ref=False):
        for branch_idx in self.storage.keys():
            for processor_name in self.storage[branch_idx].keys():
                if keep_ref and (branch_idx == 'ref'):
                    continue
                if processor_name:
                    self.storage[branch_idx][processor_name] = dict()
        
    def store(self, branch_idx, processor_name, param_name, param):
        if not 'guidance' in self.ops_dict[processor_name][param_name]:
            param = param.detach()

        if not branch_idx in list(self.storage.keys()):
            self.storage[branch_idx] = dict()
        if not processor_name in list(self.storage[branch_idx].keys()):
            self.storage[branch_idx][processor_name] = dict()
            
        if ('attn2' in processor_name) and (param_name == 'attention_probs'):
            if branch_idx == 'ref':
                if self.ref_token_ids:
                    self.storage[branch_idx][processor_name][param_name] = param[:,:,self.ref_token_ids]
            else:
                if self.all_token_ids[branch_idx]:
                    self.storage[branch_idx][processor_name][param_name] = param[:,:,self.all_token_ids[branch_idx]]
        else:
            self.storage[branch_idx][processor_name][param_name] = param 
        

    @torch.no_grad()
    def kmeans_cluster(self, batch_attn, num_clusters=6):
        # attn: [1, w*h, w*h]
        all_cluster_labels = []
        for seed in range(batch_attn.shape[0]):
            attn = batch_attn[seed].unsqueeze(dim=0)
            clustering = KMeans(n_clusters=num_clusters, num_init=self.num_kmeans_init)
            clustering_results = clustering(attn, centers=None)
            cluster_labels = clustering_results.labels
            all_cluster_labels.append(cluster_labels)
        all_cluster_labels_tensor = torch.cat(all_cluster_labels, dim=0)
        return all_cluster_labels_tensor
    
    def get_masks_from_attn(self, attn, num_clusters_list=[5]):
        # attn: [1, w*h, w*h]
        batch_masks = []
        for seed in range(attn.shape[0]):
            seed_attn = attn[seed].unsqueeze(dim=0)
            seed_masks = []
            for num_clusters in num_clusters_list:
                cluster_labels = self.kmeans_cluster(seed_attn, num_clusters=num_clusters)
                for label in range(num_clusters):
                    mask_1d = (cluster_labels == label).to(dtype=attn.dtype)   # [1, w*h]
                    factor = int(np.sqrt(mask_1d.shape[1] // (self.w_min * self.h_min)))
                    mask = mask_1d.reshape(self.h_min*factor, self.w_min*factor)
                    seed_masks.append(mask)
                del cluster_labels
                gc.collect()
                torch.cuda.empty_cache()                
            batch_masks.append(seed_masks)
                
        return batch_masks

    def choose_mask(self, batch_masks, ref_masks=None, point=None): 
        # ref_mask: [w, h]
        chosen_masks = []
        
        if ref_masks is not None:
            for seed, (seed_masks, ref_mask) in enumerate(zip(batch_masks, ref_masks)):
                max_idx = 0
                max_overlap = 0            
                for idx, mask in enumerate(seed_masks):
                    m1 = mask * ref_mask
                    m2 = mask + ref_mask
                    m2[m2>1.] = 1.
                    overlap = m1.sum() / m2.sum()
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = idx
                if max_overlap >= self.mask_overlap_threshold:
                    chosen_masks.append(seed_masks[max_idx])  
                else:
                    rect_mask = ref_mask.clone()
                    non_zero_coords = torch.nonzero(rect_mask, as_tuple=False)
                    y_min, x_min = torch.min(non_zero_coords, dim=0).values
                    y_max, x_max = torch.max(non_zero_coords, dim=0).values
                    rect_mask[y_min:y_max+1, x_min:x_max+1] = 1.  # [w, h]
                    chosen_masks.append(rect_mask)
                    # chosen_masks.append(ref_mask)
                # print('seed: ', seed, 'max_overlap: ', max_overlap.item())  
        elif point is not None:
            for seed_masks in batch_masks:
                x, y = point
                for mask in seed_masks:  
                    if mask[y, x]:
                        chosen_masks.append(mask)
                        break
        return chosen_masks
    
    def store_masks(self, branch_idx, feature_masks_tensor, prefix=""):
        bs = feature_masks_tensor.shape[0]
        n_tokens = feature_masks_tensor.shape[1]
        feature_mask_8 = (transforms.Resize((self.h_min*8, self.w_min*8))(feature_masks_tensor)).permute(0, 2, 3, 1).reshape(bs, -1, n_tokens)
        feature_mask_4 = (transforms.Resize((self.h_min*4, self.w_min*4))(feature_masks_tensor)).permute(0, 2, 3, 1).reshape(bs, -1, n_tokens)
        feature_mask_2 = (transforms.Resize((self.h_min*2, self.w_min*2))(feature_masks_tensor)).permute(0, 2, 3, 1).reshape(bs, -1, n_tokens)
        feature_mask_1 = (transforms.Resize((self.h_min, self.w_min))(feature_masks_tensor)).permute(0, 2, 3, 1).reshape(bs, -1, n_tokens)
        feature_mask_8[feature_mask_8>0.5] = 1.
        feature_mask_8[feature_mask_8<=0.5] = 0.
        feature_mask_4[feature_mask_4>0.5] = 1.
        feature_mask_4[feature_mask_4<=0.5] = 0.
        feature_mask_2[feature_mask_2>0.5] = 1.
        feature_mask_2[feature_mask_2<=0.5] = 0.
        feature_mask_1[feature_mask_1>0.5] = 1.
        feature_mask_1[feature_mask_1<=0.5] = 0.
        self.store(branch_idx, '', prefix+'feature_mask_8', feature_mask_8)
        self.store(branch_idx, '', prefix+'feature_mask_4', feature_mask_4)
        self.store(branch_idx, '', prefix+'feature_mask_2', feature_mask_2)
        self.store(branch_idx, '', prefix+'feature_mask_1', feature_mask_1)
        gc.collect()
        torch.cuda.empty_cache() 
        
    # Convert predefined mask maps to masks with different resolutions 
    def init_feature_masks(self, feature_masks=None, points=None, num_clusters=6, bs=1):   
        
        
        if feature_masks is not None:
            feature_masks_tensor = torch.cat(feature_masks, dim=0).unsqueeze(dim=0)
        else:
            
            attns = []
            for processor_name in self.ops_dict.keys():
                for param_name in self.ops_dict[processor_name].keys():
                    if 'mask' in self.ops_dict[processor_name][param_name]:
                        attn = self.extract('ref', processor_name, param_name)
                        attns.append(attn)
                        
            mean_attn = torch.stack(attns, dim=0).mean(dim=0)
            
            factor = int(np.sqrt(mean_attn.shape[1] // (self.h_min * self.w_min)))
            
            batch_masks = self.get_masks_from_attn(mean_attn, num_clusters_list=[num_clusters])
        
            feature_masks = []
            for point in points:
                x, y = point
                resized_point = (x * factor // 64, y * factor // 64)
                chosen_mask = self.choose_mask(batch_masks, point=resized_point)[0]
                feature_masks.append(chosen_mask)
                
            del batch_masks
            gc.collect()
            torch.cuda.empty_cache()

            feature_masks_tensor = torch.stack(feature_masks, dim=0).unsqueeze(dim=0) # [1, channels, h, w]
            
            
        self.store_masks('ref', feature_masks_tensor)
        
        for rid in range(feature_masks_tensor.shape[1]):
            custom_feature_mask = feature_masks_tensor[:,rid,:,:].unsqueeze(dim=1)
            self.store_masks(rid+1, custom_feature_mask.repeat(bs, 1, 1, 1))
            self.store_masks(rid+1, custom_feature_mask.repeat(bs, 1, 1, 1), prefix="custom_")
            self.store_masks(rid+1, custom_feature_mask.repeat(bs, 1, 1, 1), prefix="base_")
            
    # Get a new mask for all resolutions based on the current attention, and the mask from the previous step, not taking into account overlaps.
    def update_custom_masks(self, branch_idx): 
        assert branch_idx != 'ref'
        assert branch_idx != 0
        
        if branch_idx in list(range(len(self.all_token_ids))):
            num_clusters_min = len(self.all_token_ids)
            num_clusters_max = num_clusters_min * 2     
            n_clusters_list = list(range(num_clusters_min, num_clusters_max))       
            
            custom_attns = []
            base_attns = []
            
            for processor_name in self.ops_dict.keys():
                for param_name in self.ops_dict[processor_name].keys():
                    if 'mask' in self.ops_dict[processor_name][param_name]:
                        custom_attn = self.extract(branch_idx, processor_name, param_name)
                        custom_attns.append(custom_attn)
                        base_attn = self.extract(0, processor_name, param_name)
                        base_attns.append(base_attn)
                        
            mean_custom_attn = torch.stack(custom_attns, dim=0).mean(dim=0) 
            mean_base_attn = torch.stack(base_attns, dim=0).mean(dim=0)
            
            factor = int(np.sqrt(mean_custom_attn.shape[1] // (self.w_min*self.h_min)))            
            
            old_custom_masks_1d = self.extract(branch_idx, '', f'custom_feature_mask_{factor}')
            old_custom_masks = old_custom_masks_1d.reshape(-1, self.h_min*factor, self.w_min*factor)
            
            old_base_masks_1d = self.extract(branch_idx, '', f'base_feature_mask_{factor}')
            old_base_masks = old_base_masks_1d.reshape(-1, self.h_min*factor, self.w_min*factor)
            
            
            if not self.rect_mask:
                batch_custom_masks = self.get_masks_from_attn(mean_custom_attn, n_clusters_list)
                batch_base_masks = self.get_masks_from_attn(mean_base_attn, n_clusters_list)         
                
                chosen_custom_masks = self.choose_mask(batch_custom_masks, ref_masks=old_custom_masks)
                chosen_base_masks = self.choose_mask(batch_base_masks, ref_masks=old_base_masks)
            else:
                chosen_custom_masks = []
                for ref_mask in old_custom_masks:
                    rect_mask = ref_mask.clone()
                    non_zero_coords = torch.nonzero(rect_mask, as_tuple=False)
                    y_min, x_min = torch.min(non_zero_coords, dim=0).values
                    y_max, x_max = torch.max(non_zero_coords, dim=0).values
                    rect_mask[y_min:y_max+1, x_min:x_max+1] = 1.  # [w, h]
                    chosen_custom_masks.append(rect_mask)   
                chosen_base_masks = []
                for ref_mask in old_base_masks:
                    rect_mask = ref_mask.clone()
                    non_zero_coords = torch.nonzero(rect_mask, as_tuple=False)
                    y_min, x_min = torch.min(non_zero_coords, dim=0).values
                    y_max, x_max = torch.max(non_zero_coords, dim=0).values
                    rect_mask[y_min:y_max+1, x_min:x_max+1] = 1.  # [w, h]
                    chosen_base_masks.append(rect_mask)
           
                
            
            new_custom_masks = torch.stack(chosen_custom_masks, dim=0).unsqueeze(dim=1)
            new_base_masks = torch.stack(chosen_base_masks, dim=0).unsqueeze(dim=1)
    
            self.store_masks(branch_idx, new_custom_masks, prefix="custom_")
            self.store_masks(branch_idx, new_base_masks, prefix="base_")       
            
    # Update masks and handle overlaps          
    def refine_feature_masks(self): 
        
        for branch_idx in range(1, len(self.all_token_ids)):
            self.update_custom_masks(branch_idx)       
            
        for factor in [1, 2, 4, 8]:
            combined_masks = []
            old_combined_masks = []          
            
            for branch_idx in range(1, len(self.all_token_ids)):
                custom_mask = self.extract(branch_idx, '', f'custom_feature_mask_{factor}')
                base_mask = self.extract(branch_idx, '', f'base_feature_mask_{factor}')  
                combined_mask = custom_mask + base_mask
                combined_mask[combined_mask > 1.] = 1.
                combined_masks.append(combined_mask)
                old_combined_mask = self.extract(branch_idx, '', f'feature_mask_{factor}')
                old_combined_masks.append(old_combined_mask)
            inter_overlap_mask = torch.stack(combined_masks, dim=0).sum(dim=0)
            inter_overlap_mask[inter_overlap_mask<=1] = 0.
            inter_overlap_mask[inter_overlap_mask>1] = 1.     
            for branch_idx in range(1, len(self.all_token_ids)):
                old_combined_mask = old_combined_masks[branch_idx - 1]
                combined_mask = combined_masks[branch_idx - 1]
                refined_mask = old_combined_mask * inter_overlap_mask + combined_mask * (1 - inter_overlap_mask)
                self.store(branch_idx, '', f'feature_mask_{factor}', refined_mask)               
                
    def visualize_labels(self, batch_labels):
        bs = len(batch_labels)
        factor = int(np.sqrt(batch_labels.shape[1] // (self.h_min*self.w_min)))
        label_map_imgs = []
        for seed in range(bs):
            seed_labels = batch_labels[seed]
            seed_labels_2d = seed_labels.reshape(self.h_min*factor, self.w_min*factor).cpu().numpy()
            n_colors = seed_labels_2d.max() + 1
            colors = gen_n_colors(n_colors, seed=0, shuffle=True)
            label_map = np.zeros((seed_labels_2d.shape[0], seed_labels_2d.shape[1], 3))
            for label, color in zip(range(n_colors), colors):
                label_map[seed_labels_2d == label, :] = color
            label_map_img = Image.fromarray(label_map.astype(np.uint8))
            label_map_imgs.append(label_map_img)
        return label_map_imgs
                
    def view_self_attn(self, processors_view, self_attn_outdir, num_clusters=5):
        for processor_name in processors_view:
            ref_attn = self.extract('ref', processor_name, 'attention_probs')  # [1, h*w, h*w]
            if ref_attn is not None:
                cluster_labels = self.kmeans_cluster(ref_attn, num_clusters)
                segment_map_imgs = self.visualize_labels(cluster_labels)
                segment_map_img = segment_map_imgs[0]
                outdir = Path(self_attn_outdir) / Path(f"ref/{processor_name}")
                outdir.mkdir(exist_ok=True, parents=True)
                filename = str(outdir / Path(f"{self.step:04d}"))
                segment_map_img.save(f'{filename}.png')
            
            for branch_idx in range(len(self.all_token_ids)):
                attn = self.extract(branch_idx, processor_name, 'attention_probs')
                if attn is not None:
                    cluster_labels = self.kmeans_cluster(attn, num_clusters)
                    segment_map_imgs = self.visualize_labels(cluster_labels)
                    for seed, segment_map_img in enumerate(segment_map_imgs):
                        outdir = Path(self_attn_outdir) / Path(f"seed_{seed}/branch_{branch_idx}/{processor_name}")
                        outdir.mkdir(exist_ok=True, parents=True)
                        filename = str(outdir / Path(f"{self.step:04d}"))
                        segment_map_img.save(f'{filename}.png')
                
                
    
    def view_cross_attn(self, processors_view, cross_attn_outdir):
        for processor_name in processors_view:
            ref_attn_prob = self.extract('ref', processor_name, 'attention_probs')
            if ref_attn_prob is not None:
                ref_attn_norm = (ref_attn_prob - ref_attn_prob.min(dim=1, keepdim=True)[0]) / (ref_attn_prob.max(dim=1, keepdim=True)[0] - ref_attn_prob.min(dim=1, keepdim=True)[0] + 1e-7)
                factor = int(np.sqrt(ref_attn_norm.shape[1] // (self.h_min*self.w_min)))
                for tid, ref_token_id in enumerate(self.ref_token_ids):
                    ref_attn_map = ref_attn_norm[0,:,tid].reshape(self.h_min*factor, self.w_min*factor).cpu().numpy()
                    heatmap = cv2.applyColorMap(np.uint8(255 * ref_attn_map), cv2.COLORMAP_JET)
                    outdir = Path(cross_attn_outdir) / Path(f"ref/token_{ref_token_id}/{processor_name}")
                    outdir.mkdir(exist_ok=True, parents=True)
                    filename = str(outdir / Path(f"{self.step:04d}"))
                    cv2.imwrite(f"{filename}.png", heatmap)
            for branch_idx, token_ids in enumerate(self.all_token_ids):
                gen_attn_prob = self.extract(branch_idx, processor_name, 'attention_probs')
                if gen_attn_prob is not None:
                    gen_attn_norm = (gen_attn_prob - gen_attn_prob.min(dim=1, keepdim=True)[0]) / (gen_attn_prob.max(dim=1, keepdim=True)[0] - gen_attn_prob.min(dim=1, keepdim=True)[0] + 1e-7)
                    factor = int(np.sqrt(gen_attn_norm.shape[1] // (self.h_min*self.w_min)))
                    for seed in range(gen_attn_norm.shape[0]):
                        gen_attn_maps = []
                        for tid, token_id in enumerate(token_ids):
                            gen_attn_map = gen_attn_norm[seed,:,tid].reshape(self.h_min*factor, self.w_min*factor).cpu().numpy()
                            gen_attn_maps.append(gen_attn_map)
                            heatmap = cv2.applyColorMap(np.uint8(255 * gen_attn_map), cv2.COLORMAP_JET)
                            outdir = Path(cross_attn_outdir) / Path(f"seed_{seed}/branch_{branch_idx}/token_{token_id}/{processor_name}")
                            outdir.mkdir(exist_ok=True, parents=True)
                            filename = str(outdir / Path(f"{self.step:04d}"))
                            cv2.imwrite(f"{filename}.png", heatmap)
                        mean_gen_attn_map = np.stack(gen_attn_maps, axis=0).mean(axis=0)
                        heatmap = cv2.applyColorMap(np.uint8(255 * mean_gen_attn_map), cv2.COLORMAP_JET)
                        outdir = Path(cross_attn_outdir) / Path(f"seed_{seed}/branch_{branch_idx}/mean/{processor_name}")
                        outdir.mkdir(exist_ok=True, parents=True)
                        filename = str(outdir / Path(f"{self.step:04d}"))
                        cv2.imwrite(f"{filename}.png", heatmap)

    def view_feature_mask(self, feature_mask_outdir, prefix=""):
        if self.step == 0:
            for factor in [1, 2, 4, 8]:
                feature_masks = self.extract('ref', '', f'{prefix}feature_mask_{factor}')
                if feature_masks is not None:
                    for mid in range(feature_masks.shape[-1]):
                        feature_mask = feature_masks[0, :, mid].reshape(self.h_min*factor, self.w_min*factor)
                        mask_image = transforms.ToPILImage()(feature_mask.cpu())
                        outdir = Path(feature_mask_outdir) / Path('ref') / Path(f"factor_{factor}") / Path(f"mask_{mid}")
                        outdir.mkdir(exist_ok=True, parents=True)                        
                        mask_image.save(str(Path(outdir) / Path(f"{self.step:03d}.png")))                        


        
        for branch_idx in range(len(self.all_token_ids)):
            for factor in [1, 2, 4, 8]:
                feature_masks = self.extract(branch_idx, '', f'{prefix}feature_mask_{factor}')
                if feature_masks is not None:
                    for seed in range(feature_masks.shape[0]):
                        for mid in range(feature_masks.shape[-1]):
                            feature_mask = feature_masks[seed, :, mid].reshape(self.h_min*factor, self.w_min*factor)
                            mask_image = transforms.ToPILImage()(feature_mask.cpu())
                            outdir = Path(feature_mask_outdir) / Path(f"seed_{seed}") / Path(f"branch_{branch_idx}") / Path(f"factor_{factor}") / Path(f"mask_{mid}")
                            outdir.mkdir(exist_ok=True, parents=True)                        
                            mask_image.save(str(Path(outdir) / Path(f"{self.step:03d}.png")))
                    
                    