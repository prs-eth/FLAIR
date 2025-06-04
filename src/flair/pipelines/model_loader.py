# functions to load models from config
import numpy as np
import torch
import re
import os
import math
from diffusers import (
    BitsAndBytesConfig
)
from diffusers import AutoencoderTiny


from flair.pipelines import sd3



@torch.no_grad()
def load_sd3(config, device):
    if isinstance(device, list):
        device = device[0]
    if config["quantize"]:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        nf4_config = None
    if config["model"] == "SD3.5-large":
        pipe = sd3.SD3Wrapper.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16, quantization_config=nf4_config
        )
    elif config["model"] == "SD3.5-large-turbo":
        pipe = sd3.SD3Wrapper.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.bfloat16, quantization_config=nf4_config,
        )
    else:
        pipe = sd3.SD3Wrapper.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16, quantization_config=nf4_config)
    # maybe use tiny autoencoder
    if config["use_tiny_ae"]:
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=torch.float16)

    # encode prompts
    inp_kwargs_list = []
    prompts = config["prompt"]
    pipe._guidance_scale = config["guidance"]
    pipe._joint_attention_kwargs = {"ip_adapter_image_embeds": None}
    for prompt in prompts:
        print(f"Generating prompt embeddings for: {prompt}")
        pipe.text_encoder.to(device).to(torch.bfloat16)
        pipe.text_encoder_2.to(device).to(torch.bfloat16)
        pipe.text_encoder_3.to(device).to(torch.bfloat16)
        # encode
        (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=config["negative_prompt"],
            negative_prompt_2=config["negative_prompt"],
            negative_prompt_3=config["negative_prompt"],
            do_classifier_free_guidance=pipe.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )

        if pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        inp_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "guidance": config["guidance"],
        }
        inp_kwargs_list.append(inp_kwargs)
    pipe.vae.to(device).to(torch.bfloat16)
    pipe.transformer.to(device).to(torch.bfloat16)


    return pipe, inp_kwargs_list

def load_model(config, device=["cuda"]):
    if re.match(r"SD3*", config["model"]):
        return load_sd3(config, device)
    else:
        raise ValueError(f"Unknown model type {config['model']}")
