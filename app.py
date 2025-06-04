import gradio as gr
from gradio_imageslider import ImageSlider  # Replaces gr.ImageCompare
import torch
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import os
import sys
import json # Added import
import copy
# Add project root to sys.path to allow direct import of var_post_samp
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flair.pipelines import model_loader
from flair import var_post_samp, degradations



CONFIG_FILE_PATH = "./configs/inpainting_gradio.yaml"
DTYPE = torch.bfloat16

# Global variables to hold the model and config
MODEL = None
POSTERIOR_MODEL = None
BASE_CONFIG = None
DEVICES = None
PRIMARY_DEVICE = None
# project_root is already defined globally, will be used by save_configuration

SR_CONFIG_FILE_PATH = "./configs/x12_gradio.yaml"

# Function to save the current configuration for demo examples
def save_configuration(image_editor_data, image_input, prompt, seed_val, task, random_seed_bool, steps_val):
    global project_root # Ensure access to the globally defined project_root
    if task == "Super Resolution":
        if image_input is None:
            return gr.Markdown("""<p style='color:red;'>Error: No low-resolution image loaded.</p>""")
        # For Super Resolution, we don't need a mask, just the image
        input_image = image_input
        mask_image = None
    else:  # Inpainting task
        if image_editor_data is None or image_editor_data['background'] is None:
            return gr.Markdown("""<p style='color:red;'>Error: No background image loaded.</p>""")
        
        # Check if layers exist and the first layer (mask) is not None
        if not image_editor_data['layers'] or image_editor_data['layers'][0] is None:
            return gr.Markdown("""<p style='color:red;'>Error: No mask drawn. Please use the brush tool to draw a mask.</p>""")

        input_image = image_editor_data['background']
        mask_image = image_editor_data['layers'][0]

    metadata = {
        "prompt": prompt,
        "seed_on_slider": int(seed_val),
        "use_random_seed_checkbox": bool(random_seed_bool),
        "num_steps": int(steps_val),
        "task_type": task  # Always inpainting for now
    }

    demo_images_dir = os.path.join(project_root, "demo_images")
    try:
        os.makedirs(demo_images_dir, exist_ok=True)
    except Exception as e:
        return gr.Markdown(f"""<p style='color:red;'>Error creating directory {demo_images_dir}: {str(e)}</p>""")

    i = 0
    while True:
        base_filename = f"demo_{i}"
        meta_check_path = os.path.join(demo_images_dir, f"{base_filename}_meta.json")
        if not os.path.exists(meta_check_path):
            break
        i += 1
    
    image_save_path = os.path.join(demo_images_dir, f"{base_filename}_image.png")
    mask_save_path = os.path.join(demo_images_dir, f"{base_filename}_mask.png")
    meta_save_path = os.path.join(demo_images_dir, f"{base_filename}_meta.json")

    try:
        input_image.save(image_save_path)
        if mask_image is not None:
            # Ensure mask is saved in a usable format, e.g., 'L' mode for grayscale, or 'RGBA' if it has transparency
            if mask_image.mode != 'L' and mask_image.mode != '1': # If not already grayscale or binary
                mask_image = mask_image.convert('RGBA') # Preserve transparency if drawn, or convert to L
            mask_image.save(mask_save_path)
        
        with open(meta_save_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        return gr.Markdown(f"""<p style='color:green;'>Configuration saved as {base_filename} in demo_images folder.</p>""")
    except Exception as e:
        return gr.Markdown(f"""<p style='color:red;'>Error saving configuration: {str(e)}</p>""")

def embed_prompt(prompt, device):
    print(f"Generating prompt embeddings for: {prompt}")
    with torch.no_grad(): # Add torch.no_grad() here
        POSTERIOR_MODEL.model.text_encoder.to(device).to(torch.bfloat16)
        POSTERIOR_MODEL.model.text_encoder_2.to(device).to(torch.bfloat16)
        POSTERIOR_MODEL.model.text_encoder_3.to(device).to(torch.bfloat16)
        (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ) = POSTERIOR_MODEL.model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            do_classifier_free_guidance=POSTERIOR_MODEL.model.do_classifier_free_guidance,
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
    # POSTERIOR_MODEL.model.text_encoder.to("cpu").to(torch.bfloat16)
    # POSTERIOR_MODEL.model.text_encoder_2.to("cpu").to(torch.bfloat16)
    # POSTERIOR_MODEL.model.text_encoder_3.to("cpu").to(torch.bfloat16)
    torch.cuda.empty_cache()  # Clear GPU memory after embedding generation
    return {
        "prompt_embeds": prompt_embeds.to(device, dtype=DTYPE),
        "negative_prompt_embeds": negative_prompt_embeds.to(device, dtype=DTYPE) if negative_prompt_embeds is not None else None,
        "pooled_prompt_embeds": pooled_prompt_embeds.to(device, dtype=DTYPE),
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.to(device, dtype=DTYPE) if negative_pooled_prompt_embeds is not None else None
    }

def initialize_globals():
    global MODEL, POSTERIOR_MODEL, BASE_CONFIG, DEVICES, PRIMARY_DEVICE

    print("Global initialization started...")
    # Setup device (run once)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        DEVICES = [f"cuda:{i}" for i in range(num_gpus)]
        PRIMARY_DEVICE = DEVICES[0]
        print(f"Initializing with devices: {DEVICES}, Primary: {PRIMARY_DEVICE}")
    else:
        DEVICES = ["cpu"]
        PRIMARY_DEVICE = "cpu"
        print("No CUDA devices found. Initializing with CPU.")

    # Load base configuration (once)
    with open(CONFIG_FILE_PATH, "r") as f:
        BASE_CONFIG = yaml.safe_load(f)
    
    # Prepare a temporary config for the initial model and posterior_model loading
    init_config = BASE_CONFIG.copy()
    
    # Ensure prompt/caption settings are valid for model_loader for initialization
    # Forcing prompt mode for initial load.
    init_config["prompt"] = [BASE_CONFIG.get("prompt", "Initialization prompt")]
    init_config["caption_file"] = None
    
    # Default values that might be needed by model_loader or utils called within
    init_config.setdefault("target_file", "dummy_target.png") 
    init_config.setdefault("result_file", "dummy_results/")
    init_config.setdefault("seed", random.randint(0, 2**32 - 1)) # Init with a random seed

    print("Loading base model and variational posterior model once...")
    # MODEL is the main diffusion model, loaded once.
    # inp_kwargs_for_init are based on init_config, not directly used for subsequent inferences.
    model_obj, _ = model_loader.load_model(init_config, device=DEVICES)
    MODEL = model_obj

    # Initialize VariationalPosterior once with the loaded MODEL and init_config.
    # Its internal forward_operator will be based on init_config's degradation settings,
    # but will be replaced in each inpaint_image call.
    POSTERIOR_MODEL = var_post_samp.VariationalPosterior(MODEL, init_config)
    print("Global initialization complete.")


def load_config_for_inference(prompt_text, seed=None):
    # This function is now for creating a temporary config for each inference call,
    # primarily to get up-to-date inp_kwargs via model_loader.
    # It starts from BASE_CONFIG and applies current overrides.
    if BASE_CONFIG is None:
        raise RuntimeError("Base config not initialized. Call initialize_globals().")

    current_config = BASE_CONFIG.copy()

    current_config["prompt"] = [prompt_text] # Override with user's prompt
    current_config["caption_file"] = None # Ensure we are in prompt mode

    if seed is None:
        seed = current_config.get("seed", random.randint(0, 2**32 - 1))
    current_config["seed"] = seed
    # Set global seeds for reproducibility for the current call
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using seed for current inference: {seed}")
    
    # Ensure other necessary fields are in 'current_config' if model_loader needs them
    current_config.setdefault("target_file", "dummy_target.png") 
    current_config.setdefault("result_file", "dummy_results/")

    return current_config

def preprocess_image(pil_image, resolution, is_mask=False):
    img = pil_image.convert("RGB") if not is_mask else pil_image.convert("L")

    # Calculate new dimensions to maintain aspect ratio, making shorter edge 'resolution'
    original_width, original_height = img.size
    if original_width < original_height:
        new_short_edge = resolution
        new_long_edge = int(resolution * (original_height / original_width))
        new_width = new_short_edge
        new_height = new_long_edge
    else:
        new_short_edge = resolution
        new_long_edge = int(resolution * (original_width / original_height))
        new_height = new_short_edge
        new_width = new_long_edge

    # TF.resize expects [height, width]
    img = TF.resize(img, [new_height, new_width], interpolation=TF.InterpolationMode.LANCZOS)

    # Center crop to the target square resolution
    img = TF.center_crop(img, [resolution, resolution])

    img_tensor = TF.to_tensor(img) # Scales to [0, 1]
    if is_mask:
        # Ensure mask is binary (0 or 1), 1 for region to inpaint
        # The mask from ImageEditor is RGBA, convert to L first.
        img = img.convert('L')
        img_tensor = TF.to_tensor(img) # Recalculate tensor after convert
        img_tensor = (img_tensor == 0.) # Threshold for mask (drawn parts are usually non-black)
        img_tensor = img_tensor.repeat(3, 1, 1) # Repeat mask across 3 channels
    else:
        # Normalize image to [-1, 1]
        img_tensor = img_tensor * 2 - 1 
    return img_tensor.unsqueeze(0) # Add batch dimension

def preprocess_lr_image(pil_image, resolution, device, dtype):
    if pil_image is None:
        raise ValueError("Input PIL image cannot be None.")
    img = pil_image.convert("RGB")

    # Center crop to the target square resolution (no resizing)
    img = TF.center_crop(img, [resolution, resolution])

    img_tensor = TF.to_tensor(img)  # Scales to [0, 1]
    # Normalize image to [-1, 1]
    img_tensor = img_tensor * 2 - 1 
    return img_tensor.unsqueeze(0).to(device, dtype=dtype) # Add batch dimension and move to device


def postprocess_image(image_tensor):
    # Remove batch dimension, move to CPU, convert to float
    image_tensor = image_tensor.squeeze(0).cpu().float()
    # Denormalize from [-1, 1] to [0, 1]
    image_tensor = image_tensor * 0.5 + 0.5
    # Clip values to [0, 1]
    image_tensor = torch.clamp(image_tensor, 0, 1)
    # Convert to PIL Image
    pil_image = TF.to_pil_image(image_tensor)
    return pil_image

def inpaint_image(image_editor_output, prompt_text, fixed_seed_value, use_random_seed, guidance_scale, num_steps): # MODIFIED: seed_input changed to fixed_seed_value, use_random_seed
    try:
        if image_editor_output is None:
            raise gr.Error("Please upload an image and draw a mask.")

        input_pil = image_editor_output['background'] 
        
        if not image_editor_output['layers'] or image_editor_output['layers'][0] is None:
            raise gr.Error("Please draw a mask on the image using the brush tool.")
        mask_pil = image_editor_output['layers'][0]


        if input_pil is None:
            raise gr.Error("Please upload an image.")
        if mask_pil is None: 
            raise gr.Error("Please draw a mask on the image.")

        current_seed = None
        if use_random_seed:
            current_seed = None # load_config_for_inference will generate a random seed
        else:
            try:
                current_seed = int(fixed_seed_value)
            except ValueError:
                # This should ideally not happen with a slider, but good for robustness
                raise gr.Error("Seed must be an integer.")

        # Prepare config for current inference (gets prompt, seed)
        current_config = load_config_for_inference(prompt_text, current_seed)
        resolution = current_config["resolution"]

        # MODIFIED: Set num_steps from slider into the current_config
        # Assuming 'num_steps' is a key POSTERIOR_MODEL will use from its config.
        # Common alternatives could be current_config['solver_kwargs']['n_steps'] = num_steps
        current_config['n_steps'] = int(num_steps) 
        print(f"Using num_steps: {current_config['n_steps']}")


        # Preprocess image and mask
        guidance_img_tensor = preprocess_image(input_pil, resolution, is_mask=False).to(PRIMARY_DEVICE, dtype=DTYPE)
        # Mask from ImageEditor is RGBA, preprocess_image will handle conversion to L and then binary
        mask_tensor = preprocess_image(mask_pil, resolution, is_mask=True).to(PRIMARY_DEVICE, dtype=DTYPE) 
        
        # Get inp_kwargs for the CURRENT prompt and config.
        print("Preparing inference inputs (e.g., prompt embeddings)...")
        prompt_embeds = embed_prompt(prompt_text, device=PRIMARY_DEVICE) # Embed the prompt for the current inference
        current_inp_kwargs = prompt_embeds
        # MODIFIED: Use guidance_scale from slider
        current_inp_kwargs['guidance'] = float(guidance_scale) 
        print(f"Using guidance_scale: {current_inp_kwargs['guidance']}")
        
        # Update the global POSTERIOR_MODEL's config for this call.
        # This ensures its methods use the latest settings (like num_steps) if they access self.config.
        POSTERIOR_MODEL.config = current_config
        POSTERIOR_MODEL.model._guidance_scale = guidance_scale
        print("Applying forward operator (masking)...")
        # Directly set the forward_operator on the global POSTERIOR_MODEL instance
        # H and W are height and width of the guidance image tensor
        POSTERIOR_MODEL.forward_operator = degradations.Inpainting(
            mask=mask_tensor.bool()[0], # Inpainting often expects a boolean mask
            H=guidance_img_tensor.shape[2], 
            W=guidance_img_tensor.shape[3], 
            noise_std=0, 
        )
        y = POSTERIOR_MODEL.forward_operator(guidance_img_tensor)    
        
        print("Running inference...")
        with torch.no_grad():
            # Use the global POSTERIOR_MODEL instance
            result_dict = POSTERIOR_MODEL.forward(y, current_inp_kwargs)
        
        x_hat = result_dict["x_hat"]
        
        print("Postprocessing result...")
        output_pil = postprocess_image(x_hat)
        
        # Convert mask tensor to PIL image for display
        # Mask tensor is [0, 1], take one channel, convert to PIL
        mask_display_tensor = mask_tensor.squeeze(0).cpu().float() # Remove batch, move to CPU
        # If mask_tensor was (B, 3, H, W) and binary 0 or 1 (after repeat)
        # We can take any channel, e.g., mask_display_tensor[0]
        # Ensure it's (H, W) or (1, H, W) for to_pil_image
        if mask_display_tensor.ndim == 3 and mask_display_tensor.shape[0] == 3: # (C, H, W)
            mask_display_tensor = mask_display_tensor[0] # Take one channel (H, W)
        
        # Ensure it's in the range [0, 1] and suitable for PIL conversion
        # If it was 0. for masked and 1. for unmasked (or vice-versa depending on logic)
        # TF.to_pil_image expects [0,1] for single channel float
        mask_pil_display = TF.to_pil_image(mask_display_tensor)

        return output_pil, [output_pil, output_pil], current_config["seed"] # MODIFIED: Removed mask_pil_display
    except gr.Error as e: # Handle Gradio-specific errors first
        raise
    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback # Ensure traceback is imported here if not globally
        traceback.print_exc()
        # Return a more user-friendly error message to Gradio
        raise gr.Error(f"An error occurred: {str(e)}. Check console for details.")

def super_resolution_image(lr_image, prompt_text, fixed_seed_value, use_random_seed, guidance_scale, num_steps, sr_scale_factor, downscale_input):
    try:
        if lr_image is None:
            raise gr.Error("Please upload a low-resolution image.")

        current_seed = None
        if use_random_seed:
            current_seed = random.randint(0, 2**32 - 1)
        else:
            try:
                current_seed = int(fixed_seed_value)
            except ValueError:
                raise gr.Error("Seed must be an integer.")

        # Load Super-Resolution specific configuration
        if not os.path.exists(SR_CONFIG_FILE_PATH):
            raise gr.Error(f"Super-resolution config file not found: {SR_CONFIG_FILE_PATH}")
        with open(SR_CONFIG_FILE_PATH, "r") as f:
            sr_base_config = yaml.safe_load(f)

        current_sr_config = copy.deepcopy(sr_base_config) # Start with a copy of the base SR config
        current_sr_config["prompt"] = [prompt_text]
        current_sr_config["caption_file"] = None # Ensure prompt mode
        current_sr_config["seed"] = current_seed
        
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        print(f"Using seed for SR inference: {current_seed}")

        current_sr_config['n_steps'] = int(num_steps)
        current_sr_config["degradation"]["kwargs"]["scale"] = sr_scale_factor
        current_sr_config["optimizer_dataterm"]["kwargs"]["lr"] = sr_base_config.get("optimizer_dataterm", {}).get("kwargs", {}).get("lr") * sr_scale_factor**2 / (sr_base_config.get("degradation", {}).get("kwargs", {}).get("scale")**2)
        print(f"Using num_steps for SR: {current_sr_config['n_steps']}")

        # Determine target HR resolution for the output
        hr_resolution = current_sr_config.get("degradation", {}).get("kwargs", {}).get("img_size")
        # Calculate target LR dimensions based on the chosen scale factor
        target_lr_width = int(hr_resolution / sr_scale_factor)
        target_lr_height = int(hr_resolution / sr_scale_factor)
        print(f"Target LR dimensions for SR: {target_lr_width}x{target_lr_height} for scale x{sr_scale_factor}")

        print("Preparing SR inference inputs (prompt embeddings)...")
        prompt_embeds = embed_prompt(prompt_text, device=PRIMARY_DEVICE)
        current_inp_kwargs = prompt_embeds
        current_inp_kwargs['guidance'] = float(guidance_scale)
        print(f"Using guidance_scale for SR: {current_inp_kwargs['guidance']}")

        POSTERIOR_MODEL.config = current_sr_config
        POSTERIOR_MODEL.model._guidance_scale = float(guidance_scale)

        print("Applying SR forward operator...")
        
        POSTERIOR_MODEL.forward_operator = degradations.SuperResGradio(
            **current_sr_config["degradation"]["kwargs"]
        )
        
        if downscale_input:
            y_tensor = preprocess_lr_image(lr_image, hr_resolution, PRIMARY_DEVICE, DTYPE)
            # y_tensor = POSTERIOR_MODEL.forward_operator(y_tensor)
            y_tensor = torch.nn.functional.interpolate(y_tensor, scale_factor=1/sr_scale_factor, mode='bilinear', align_corners=False, antialias=True)
            # simulate 8bit input by quantizing to 8-bit
            y_tensor = ((y_tensor * 127.5 + 127.5).clamp(0, 255).to(torch.uint8) / 127.5 - 1.0).to(DTYPE) 
        else:
            # check if the input image has the correct dimensions
            if lr_image.size[0] != target_lr_width or lr_image.size[1] != target_lr_height:
                raise gr.Error(f"Input image must be {target_lr_width}x{target_lr_height} pixels for the selected scale factor of {sr_scale_factor}.")
            y_tensor = preprocess_lr_image(lr_image, target_lr_width, PRIMARY_DEVICE, DTYPE)
            # add some noise to the input image
            noise_std = current_sr_config.get("degradation", {}).get("kwargs", {}).get("noise_std", 0.0)
            y_tensor += torch.randn_like(y_tensor) * noise_std
            # save for debugging purposes
            # first convert to PIL
            pil_y = postprocess_image(y_tensor)# Remove batch dimension and convert to PIL
            pil_y.save("debug_input_image.png")  # Save the input image for debugging
                
        
        print("Running SR inference...")
        with torch.no_grad():
            result_dict = POSTERIOR_MODEL.forward(y_tensor, current_inp_kwargs)
        
        x_hat = result_dict["x_hat"]
        
        print("Postprocessing SR result...")
        output_pil = postprocess_image(x_hat)

        # Upscale input image with nearest neighbor for comparison
        upscaled_input = y_tensor.reshape(1,3,target_lr_height, target_lr_width)
        upscaled_input = POSTERIOR_MODEL.forward_operator.nn(upscaled_input)  # Use nearest neighbor upscaling
        upscaled_input = postprocess_image(upscaled_input)
        # save for debugging purposes
        upscaled_input.save("debug_upscaled_input.png")  # Save the upscaled input image for debugging
        # upscaled_input = upscaled_input.resize((hr_resolution, hr_resolution), resample=Image.NEAREST)
        return (upscaled_input, output_pil), current_sr_config["seed"]

    except gr.Error as e:
        raise
    except Exception as e:
        print(f"Error during super-resolution: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An error occurred during super-resolution: {str(e)}. Check console for details.")


# Input for seed, allowing users to set it or leave it blank for random/config default
# Determine default num_steps from BASE_CONFIG if available
default_num_steps = 50 # Fallback default
if BASE_CONFIG is not None: # Check if BASE_CONFIG has been initialized
    default_num_steps = BASE_CONFIG.get("num_steps", BASE_CONFIG.get("solver_kwargs", {}).get("num_steps", 50))

def superres_preview_preprocess(pil_image, resolution=768):
    if pil_image is None:
        return None
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    # check if image is smaller than resolution
    original_width, original_height = pil_image.size
    if original_width < resolution or original_height < resolution:
        return pil_image  # No resizing needed, return original image
    else:
        pil_image = TF.center_crop(pil_image, [resolution, resolution])
    return pil_image


# Dynamically load examples from demo_images directory
example_list_inp = []
example_list_sr = []
demo_images_dir = os.path.join(project_root, "static/demo_images")

if os.path.exists(demo_images_dir):
    filenames = sorted(os.listdir(demo_images_dir))
    processed_bases = set()
    for filename in filenames:
        if filename.startswith("demo_") and filename.endswith("_meta.json"):
            base_name = filename[:-len("_meta.json")] # e.g., "demo_0"
            if base_name in processed_bases:
                continue

            meta_path = os.path.join(demo_images_dir, filename)
            image_filename = f"{base_name}_image.png"
            image_path = os.path.join(demo_images_dir, image_filename)
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(demo_images_dir, mask_filename)

            if os.path.exists(image_path):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    task = metadata.get("task_type")
                    prompt = metadata.get("prompt", "")
                    if task == "Super Resolution":
                        example_list_sr.append([image_path, prompt, task])
                    else:
                        image_editor_input = {
                            "background": image_path,
                            "layers": [mask_path],
                            "composite": None  # Add this key to satisfy ImageEditor's as_example processing
                        }
                        example_list_inp.append([image_editor_input, prompt, task])
                    
                    # Structure for ImageEditor: { "background": filepath, "layers": [filepath], "composite": None }
                    
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {meta_path}. Skipping example {base_name}.")
                except Exception as e:
                    print(f"Warning: Error processing example {base_name}: {e}. Skipping.")
            else:
                missing_files = []
                if not os.path.exists(image_path):
                    missing_files.append(image_filename)
                if not os.path.exists(mask_path):
                    missing_files.append(mask_filename)
                print(f"Warning: Missing files for example {base_name} ({', '.join(missing_files)}). Skipping.")
else:
    print(f"Info: 'demo_images' directory not found at {demo_images_dir}. No dynamic examples will be loaded.")


if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"ERROR: Configuration file not found at {CONFIG_FILE_PATH}")
        sys.exit(1)
    
    initialize_globals()
    
    if MODEL is None or POSTERIOR_MODEL is None:
        print("ERROR: Global model initialization failed.")
        sys.exit(1)

    # --- Define Gradio UI using gr.Blocks after globals are initialized ---
    title_str = "Solving Inverse Problems with FLAIR: Inpainting Demo"
    description_str = """
Select a task (Inpainting or Super Resolution) and upload an image.
For Inpainting, draw a mask on the image to specify the area to be filled. We observed that our model can event solve simple editing task, if provided with an appropriate prompt. For large masks the step size might need to be adjusted to e.g. 80.
For Super Resolution, upload a low-resolution image and select the upscaling factor. Images are always upscaled to 768x768 pixels. Therefore, for x12 superresolution, the input image must be 64x64 pixels. You can also upload a high resolution image which will be downscaled to the correct input size. 
Use the slider to compare the low resolution input image with the super-resolved output.

"""

    # Determine default values now that BASE_CONFIG is initialized
    default_num_steps = BASE_CONFIG.get("num_steps", BASE_CONFIG.get("solver_kwargs", {}).get("num_steps", 50))
    default_guidance_scale = BASE_CONFIG.get("guidance", 2.0)

    with gr.Blocks() as iface:
        gr.Markdown(f"## {title_str}")
        gr.Markdown(description_str)

        task_selector = gr.Dropdown(
            choices=["Inpainting", "Super Resolution"],
            value="Inpainting",
            label="Task"
        )

        with gr.Row():
            with gr.Column(scale=1):  # Input column
                # Inpainting Inputs
                image_editor = gr.ImageEditor(
                    type="pil",
                    label="Upload Image & Draw Mask (for Inpainting)",
                    sources=["upload"],
                    height=512,
                    width=512,
                    visible=True
                )

                # Super Resolution Inputs
                image_input = gr.Image(
                    type="pil",
                    label="Upload Low-Resolution Image (for Super Resolution)",
                    visible=False
                )

                sr_scale_slider = gr.Dropdown(
                    choices=[2, 4, 8, 12, 24],
                    value=12,
                    label="Upscaling Factor (Super Resolution)",
                    interactive=True,
                    visible=False # Initially hidden
                )
                downscale_input = gr.Checkbox(
                    label="Downscale the provided image.",
                    value=True,
                    interactive=True,
                    visible=False # Initially hidden
                )

                # Common Inputs
                prompt_text = gr.Textbox(
                    label="Prompt",
                    placeholder="E.g., a beautiful landscape, a detailed portrait"
                )
                seed_slider = gr.Slider(
                    minimum=0,
                    maximum=2**32 -1, # Max for torch.manual_seed
                    step=1,
                    label="Seed (if not random)",
                    value=42,
                    interactive=True
                )
                
                use_random_seed_checkbox = gr.Checkbox(
                    label="Use Random Seed",
                    value=True,
                    interactive=True
                )
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=default_guidance_scale,
                    label="Guidance Scale"
                )
                num_steps_slider = gr.Slider(
                    minimum=28,
                    maximum=150,
                    step=1,
                    value=default_num_steps,
                    label="Number of Steps"
                )
                submit_button = gr.Button("Submit")

                # # Add Save Configuration button and status text
                # gr.Markdown("---") # Separator
                # save_button = gr.Button("Save Current Configuration for Demo")
                # save_status_text = gr.Markdown()

            with gr.Column(scale=1):  # Output column
                output_image_display = gr.Image(type="pil", label="Result")
                sr_compare_display = ImageSlider(label="Super-Resolution: Input vs Output", visible=False, position=0.5)


        

        # --- Task routing and visibility logic ---
        def update_visibility(task):
            is_inpainting = task == "Inpainting"
            is_super_resolution = task == "Super Resolution"
            return {
                image_editor: gr.update(visible=is_inpainting),
                image_input: gr.update(visible=is_super_resolution),
                sr_scale_slider: gr.update(visible=is_super_resolution),
                downscale_input: gr.update(visible=is_super_resolution),
                output_image_display: gr.update(visible=is_inpainting),
                sr_compare_display: gr.update(visible=is_super_resolution, position=0.5),
                downscale_input: gr.update(visible=is_super_resolution),
            }

        task_selector.change(
            fn=update_visibility,
            inputs=[task_selector],
            outputs=[image_editor, image_input, sr_scale_slider, downscale_input, output_image_display, sr_compare_display]
        )


        # MODIFIED route_task to accept sr_scale_factor
        def route_task(task, image_editor_data, lr_image_for_sr, prompt_text, fixed_seed_value, use_random_seed, guidance_scale, num_steps, sr_scale_factor_value, downscale_input):
            if task == "Inpainting":
                return inpaint_image(image_editor_data, prompt_text, fixed_seed_value, use_random_seed, guidance_scale, num_steps)
            elif task == "Super Resolution":
                result_images, seed_val = super_resolution_image(
                    lr_image_for_sr, prompt_text, fixed_seed_value, use_random_seed,
                    guidance_scale, num_steps, sr_scale_factor_value, downscale_input
                )
                return result_images[1], gr.update(value=result_images, position=0.5), seed_val
            else:
                raise gr.Error("Unsupported task.")

        submit_button.click(
            fn=route_task,
            inputs=[
                task_selector,
                image_editor,
                image_input,
                prompt_text,
                seed_slider,
                use_random_seed_checkbox,
                guidance_scale_slider,
                num_steps_slider,
                sr_scale_slider,
                downscale_input,
            ],
            outputs=[
                output_image_display,
                sr_compare_display,
                seed_slider
            ]
        )

        # Wire up the save button
        # save_button.click(
        #     fn=save_configuration,
        #     inputs=[
        #         image_editor,
        #         image_input,
        #         prompt_text,
        #         seed_slider,
        #         task_selector,
        #         use_random_seed_checkbox,
        #         num_steps_slider,
        #     ],
        #     outputs=[save_status_text]
        # )


        gr.Markdown("---") # Separator
        gr.Markdown("### Click an example to load:")
        def load_example(example_data, prompt, task):
            image_editor_input = example_data[0]
            prompt_value = example_data[1]
            if task == "Inpainting":
                image_editor.clear()  # Clear current image and mask
                if image_editor_input and image_editor_input.get("background"):
                    image_editor.upload_image(image_editor_input["background"])
                if image_editor_input and image_editor_input.get("layers"):
                    for layer in image_editor_input["layers"]:
                        image_editor.upload_mask(layer)
            elif task == "Super Resolution":
                image_input.clear()
                image_input.upload_image(image_editor_input)
                
            # Set the prompt
            prompt_text.value = prompt_value
            # Optionally, set a random seed and guidance scale
            seed_slider.value = random.randint(0, 2**32 - 1)
            guidance_scale_slider.value = default_guidance_scale
            # Set the task selector from the example
            task_selector.set_value(task)
            update_visibility(task)  # Update visibility based on task

        with gr.Row():
            gr.Examples(
                examples=example_list_sr,
                inputs=[image_input, prompt_text, task_selector],
                label="Super Resolution Examples",
                fn=load_example,
            )
        with gr.Row():
            gr.Examples(
                examples=example_list_inp,
                inputs=[image_editor, prompt_text, task_selector],
                label="Inpainting Examples",
                fn=load_example,
            )

    # --- End of Gradio UI definition ---

    print("Launching Gradio demo...")
    iface.launch()
