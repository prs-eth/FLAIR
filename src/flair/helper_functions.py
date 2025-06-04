import os
import yaml
import click
import subprocess

def parse_click_context(ctx):
    """Parse additional arguments passed via Click context."""
    extra_args = {}
    for arg in ctx.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key.startswith("--"):
                key = key[2:]  # Remove leading "--"
            extra_args[key] = yaml.safe_load(value)
    return extra_args

def generate_captions_with_seesr(pseudo_inv_dir, output_caption_file):
    """Generate captions using the SEESR model."""
    command = [
        "conda",
        "run",
        "-n",
        "seesr",
        "python",
        "/home/erbachj/scratch2/projects/var_post_samp/scripts/generate_caption.py",
        "--input_dir",
        pseudo_inv_dir,
        "--output_file",
        output_caption_file,  # Corrected argument name
    ]
    subprocess.run(command, check=True)