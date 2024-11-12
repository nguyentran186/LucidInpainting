from audioop import mul
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DiffusionPipeline, DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, \
                      EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, ControlNetModel, \
                      DDIMInverseScheduler, UNet2DConditionModel, StableDiffusionXLInpaintPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
import os
import random

import torchvision.transforms as T
# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

from .sd_step import *

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None, 
                 ddim_inv=False, use_control_net=False, textual_inversion_path = None, 
                 LoRA_path = None, guidance_opt=None):
        super().__init__()

        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        assert model_key is not None

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = "stabilityai/stable-diffusion-v1-5" if guidance_opt.base_model_key is None else guidance_opt.base_model_key # for finetuned model only

        if is_safe_tensor:
            pipe = StableDiffusionXLInpaintPipeline.from_single_file(model_key, use_safetensors=True, torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if use_control_net:
            controlnet_model_key = guidance_opt.controlnet_model_key
            self.controlnet_depth = ControlNetModel.from_pretrained(controlnet_model_key,torch_dtype=self.precision_t).to(device)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))
        
        if LoRA_path is not None:
            from lora_diffusion import tune_lora_scale, patch_pipe
            print("load lora in:.{}".format(LoRA_path))
            patch_pipe(
                pipe,
                LoRA_path,
                patch_text=True,
                patch_ti=True,
                patch_unet=True,
            )
            tune_lora_scale(pipe.unet, 1.00)
            tune_lora_scale(pipe.text_encoder, 1.00)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps        
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)

        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)
        self.generator = self.noise_gen

        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.rgb_latent_factors = torch.tensor([
                    # R       G       B
                    [ 0.298,  0.207,  0.208],
                    [ 0.187,  0.286,  0.173],
                    [-0.158,  0.189,  0.264],
                    [-0.184, -0.271, -0.473]
                ], device=self.device)
        

        print(f'[INFO] loaded stable diffusion!')

    def augmentation(self, *tensors):
        augs = T.Compose([
                        T.RandomHorizontalFlip(p=0.5),
                    ])
        
        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, mask, mask_img, noise, 
                           ind_t, ind_prev_t, 
                           text_embeddings=None, cfg=1.0, 
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0,
                           added_cond_kwargs=None):

        unet = self.unet
        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = torch.cat([cur_noisy_lat] * 2) if self.do_classifier_free_guidance else cur_noisy_lat
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat_, self.timesteps[cur_ind_t]).to(self.precision_t)
            
            cur_noisy_lat_ = torch.cat([cur_noisy_lat_, mask, mask_img], dim=1)

            timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)

            unet_output = unet(cur_noisy_lat_, 
                            timestep_model_input, 
                            encoder_hidden_states=text_embeddings,
                            added_cond_kwargs=added_cond_kwargs,
                            ).sample
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = unet_output.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
           
            pred_scores.append((cur_ind_t, noise_pred))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = self.sche_func(self.scheduler, noise_pred, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del noise_pred
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break
            
        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]


    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height//self.pipe.vae_scale_factor, width//self.pipe.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        if isinstance(generator, list):
            masked_image_latents = [
                self.vae.encode(masked_image[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
        else:
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def train_step_xl(self, prompt, negative_prompt, image, mask_image,
                    grad_scale=1, warm_up_rate = 0,
                    resolution=(1024, 1024), guidance_opt=None):
        
        # origin = (pred_rgb * (1-mask_image)).detach()
        # masked_pred_rgb = pred_rgb * mask_image.clone() + origin
        height = resolution[0]
        width = resolution[1]
        batch_size = 1
        device = self.device
        
        self.do_classifier_free_guidance = True

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        
        
        ## Time step setup
        if guidance_opt.annealing_intervals:
            current_delta_t =  int(guidance_opt.delta_t + (warm_up_rate)*(guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t =  guidance_opt.delta_t

        ind_t = torch.randint(self.min_step, self.max_step + int(self.warmup_step*warm_up_rate), (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)   
             
        ## Prepare image and mask
        original_image = image
        init_image = self.pipe.image_processor.preprocess(
            image, height=height, width=width, crops_coords=None, resize_mode="default")
        init_image = init_image.to(dtype=torch.float16)
        
        mask = self.pipe.image_processor.preprocess(
            mask_image, height=height, width=width, crops_coords=None, resize_mode="default")
        
        masked_image = init_image * (mask < 0.5)    
        
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4 
        
        ## Prepare latent and noise
        latents = self.pipe._encode_vae_image(init_image, generator=self.noise_gen)
        shape = (latents.shape)
        noise = torch.randn(shape, dtype=latents.dtype, device=latents.device, generator=self.noise_gen)
        
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size,
            height,
            width,
            prompt_embeds.dtype,
            device,
            self.generator,
            self.do_classifier_free_guidance,
        )  
        # Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
            
        height, width = latents.shape[-2:]
        height = height * self.pipe.vae_scale_factor
        width = width * self.pipe.vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)

        # 10. Prepare added time ids & embeddings
        negative_original_size = original_size
        negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self.pipe._get_add_time_ids(
            original_size,
            (0,0),
            target_size,
            6.0,
            2.5,
            negative_original_size,
            (0,0),
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        
        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)
        
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        
        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # Step 1: sample x_s with larger steps
            xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
            xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(np.ceil(ind_prev_t / xs_delta_t))
            starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

            ## To Unet
            _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, mask, masked_image_latents, noise, ind_prev_t, starting_ind, prompt_embeds, 
                                                                            guidance_opt.denoise_guidance_scale, xs_delta_t, xs_inv_steps, eta=guidance_opt.xs_eta, 
                                                                            added_cond_kwargs=added_cond_kwargs)
            # Step 2: sample x_t
            _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, mask, masked_image_latents, noise, ind_t, ind_prev_t, prompt_embeds, 
                                                                        guidance_opt.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True, 
                                                                        added_cond_kwargs=added_cond_kwargs)        

            pred_scores = pred_scores_xt + pred_scores_xs
            target = pred_scores[0][1]
        
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            unet_output = self.unet(latent_model_input.to(self.precision_t), 
                                    tt.to(self.precision_t),
                                    encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=added_cond_kwargs,
                                    ).sample
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = unet_output.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond) 

        # Define weighting function for the noise scaling
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)

        # Only calculate the gradient for masked regions
        # grad will be zero in areas where mask is zero
        grad = w(self.alphas[t]) * (noise_pred - target)

        # Apply the mask to only calculate gradients in the masked region
        # Multiplying by mask ensures that only masked regions are affected
        masked_grad = grad * mask.chunk(2)[0]  # Use the mask chunk for the input batch

        # Scale gradients as needed, and handle NaN values
        masked_grad = torch.nan_to_num(grad_scale * masked_grad)

        # Create the loss with SpecifyGradient but only apply masked_grad
        # This will effectively ignore non-masked regions in backpropagation
        loss = SpecifyGradient.apply(latents, masked_grad)

        return loss


    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)
