# Copyright (c) 2024 <Julius Erbach ETH Zurich>
#
# This file is part of the var_post_samp project and is licensed under the MIT License.
# See the LICENSE file in the project root for more information.
"""
Usage:
    python run_image_inv.py --config <config.yaml>
"""

import os
import sys
import time
import csv
import yaml
import torch
import random
import click
import numpy as np
import tqdm
import datetime
import torchvision

from flair.helper_functions import parse_click_context
from flair.pipelines import model_loader
from flair.utils import data_utils
from flair import var_post_samp


dtype = torch.bfloat16

num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    primary_device = devices[0]
    print(f"Using devices: {devices}")
    print(f"Primary device for operations: {primary_device}")
else:
    print("No CUDA devices found. Using CPU.")
    devices = ["cpu"]
    primary_device = "cpu"


@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--config", "config_file_arg", type=click.Path(exists=True), help="Path to the config file")
@click.option("--target_file", type=click.Path(exists=True), help="Path to the target file or folder")
@click.option("--result_folder", type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True), help="Path to the output folder. It will be created if it doesn't exist.")
@click.option("--mask_file", type=click.Path(exists=True), default=None, help="Path to the mask file npy. Optional used for image inpainting. True pixels are observed.")
@click.pass_context
def main(ctx, config_file_arg, target_file, result_folder, mask_file=None):
    """Main entry point for image inversion and sampling.

    The user must provide either a caption_file (with per-image captions) OR a single prompt for all images in the config YAML file.
    """
    with open(config_file_arg, "r") as f:
        config = yaml.safe_load(f)
    ctx = parse_click_context(ctx)
    config.update(ctx)
    # Read caption_file and prompt from config
    caption_file = config.get("caption_file", None)
    prompt = config.get("prompt", None)

    # Enforce mutually exclusive caption_file or prompt
    if (not caption_file and not prompt) or (caption_file and prompt):
        raise ValueError("You must provide either 'caption_file' OR 'prompt' (not both) in the config file. See documentation.")

    # wandb removed, so config_dict is just a copy
    config_dict = dict(config)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Use config values as-is (no to_absolute_path)
    caption_file = caption_file if caption_file else None

    guidance_img_iterator = data_utils.yield_images(
        target_file, size=config["resolution"]
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    name = f'results_{config["model"]}_{config["degradation"]["name"]}_resolution_{config["resolution"]}_noise_{config["degradation"]["kwargs"]["noise_std"]}_{timestamp}'
    candidate = os.path.join(name)
    while os.path.exists(candidate):
        candidate = os.path.join(f"{name}_{counter}")
        counter += 1
    output_folders = data_utils.generate_output_structure(
        result_folder,
        [
            candidate,
            f'input_{config["degradation"]["name"]}_resolution_{config["resolution"]}_noise_{config["degradation"]["kwargs"]["noise_std"]}',
            f'target_{config["degradation"]["name"]}_resolution_{config["resolution"]}_noise_{config["degradation"]["kwargs"]["noise_std"]}',
        ],
    )
    config_out = os.path.join(os.path.split(output_folders[0])[0], "config.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(config_dict, f)

    source_files = list(data_utils.find_files(target_file, ext="png"))
    num_images = len(source_files)
    print(f"Found {num_images} images.")

    # Load captions
    if caption_file:
        captions = data_utils.load_captions_from_file(caption_file, user_prompt="")
        if not captions:
            sys.exit("Error: No captions were loaded from the provided caption file.")
        if len(captions) != num_images:
            print("Warning: Number of captions does not match number of images.")
        prompts_in_order = [captions.get(os.path.basename(f), "") for f in source_files]
    else:
        # Use the single prompt for all images
        prompts_in_order = [prompt for _ in range(num_images)]

    if any(p == "" for p in prompts_in_order):
        print("Warning: Some images might not have corresponding captions or prompt is empty.")
    config["prompt"] = prompts_in_order

    model, inp_kwargs = model_loader.load_model(config, device=devices)
    if mask_file and config["degradation"]["name"] == "Inpainting":
        config["degradation"]["kwargs"]["mask"] = mask_file
    posterior_model = var_post_samp.VariationalPosterior(model, config)
    guidance_img_iterator = data_utils.yield_images(
        target_file, size=config["resolution"]
    )
    for idx, guidance_img in tqdm.tqdm(enumerate(guidance_img_iterator), total=num_images):
        guidance_img = guidance_img.to(dtype).cuda()
        y = posterior_model.forward_operator(guidance_img)
        tic = time.time()
        with torch.no_grad():
            result_dict = posterior_model.forward(y, inp_kwargs[idx])
            x_hat = result_dict["x_hat"]
        toc = time.time()
        print(f"Runtime: {toc - tic}")
        guidance_img = guidance_img.cuda()
        result_file = output_folders[0].format(idx)
        input_file = output_folders[1].format(idx)
        ground_truth_file = output_folders[2].format(idx)
        x_hat_pil = torchvision.transforms.ToPILImage()(
            x_hat.float()[0].clip(-1, 1) * 0.5 + 0.5
        )
        x_hat_pil.save(result_file)
        try:
            if config["degradation"]["name"] == "SuperRes":
                input_img = posterior_model.forward_operator.nn(y)
            else:
                input_img = posterior_model.forward_operator.pseudo_inv(y)
            input_img_pil = torchvision.transforms.ToPILImage()(
                input_img.float()[0].clip(-1, 1) * 0.5 + 0.5
            )
            input_img_pil.save(input_file)
        except Exception:
            print("Error in pseudo-inverse operation. Skipping input image save.")
        guidance_img_pil = torchvision.transforms.ToPILImage()(
            guidance_img.float()[0] * 0.5 + 0.5
        )
        guidance_img_pil.save(ground_truth_file)


if __name__ == "__main__":
    main()
