import torch
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b
import torch.nn as nn
import numpy as np
import torch.quantization


def load_pruned_state_dict(model, state_dict):
    try:
        new_state_dict = {}
        for key, value in state_dict.items():
            if '_orig' in key:
                # Handle pruned weights
                mask_key = key.replace('_orig', '_mask')
                if mask_key in state_dict:
                    pruned_weights = state_dict[key] * state_dict[mask_key]
                    new_key = key.replace('_orig', '')
                    new_state_dict[new_key] = pruned_weights
            elif '_mask' not in key:
                # This handles unpruned weights or any other entries normally
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        model.eval()  # Set model to evaluation mode after loading
    except:
        load_pruned_state_dict_weird(model, state_dict)


def load_pruned_state_dict_weird(model, state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key  # Start with assuming no change to key

        # Handle pruned weights
        if '_orig' in key:
            mask_key = key.replace('_orig', '_mask')
            if mask_key in state_dict:
                pruned_weights = state_dict[key] * state_dict[mask_key]
                new_key = key.replace('_orig', '')

        # Adjust keys for layers that are expected to contain 'linear'
        # This specifically checks for layer names in the xyz layers
        layer_indicators = ["layers_xyz.", "layer1.", "layers_dir.", "fc_"]
        if any(indicator in new_key for indicator in layer_indicators):
            # Check if 'linear' needs to be added
            if 'linear' not in new_key:
                parts = new_key.split('.')
                # Insert 'linear' before the last part (weight or bias)
                parts.insert(-1, 'linear')
                new_key = '.'.join(parts)

        # Assign the possibly adjusted weights or the original ones
        new_state_dict[new_key] = value if '_orig' not in key else pruned_weights

    # Attempt to load the modified state dictionary into the model
    try:
        model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore non-matching keys
        model.half()
        model.eval()  # Set model to evaluation mode after loading
        print("Model loaded successfully with adjusted keys.")
    except RuntimeError as e:
        print(f"Failed to load model: {e}")

def visualize(config):
    data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
        Nbits=16,  # Add Nbits parameter for quantization
        symmetric=False  # Add symmetric parameter for quantizationNbits=None,  # Add Nbits parameter for quantization
    )
    load_pruned_state_dict(model, torch.load(config.model_weight_path))

    # model = torch.
    model.eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    
    
    for name, param in model.named_parameters():
        print(name, param.dtype)
    
    exempt_layer = ['rgb_net0.0', 'rgb_net1.0', 'final_density.0', 'final_rgb.0']

    for name,layer in model.named_modules():
        if (isinstance(layer, nn.Linear) and name not in exempt_layer):
            # change q value

            # Optional: Check the sparsity you achieve in each layer
            # Convert the weight of "layer" to numpy array
            np_weight = layer.weight.data.cpu().detach().numpy()
        # Count number of zeros
            zeros = np.count_nonzero(np_weight==0)
        # Count number of parameters
        # if isinstance(layer, nn.Conv2d):
        #   total = (np_weight.shape[3]*np_weight.shape[2]*np_weight.shape[1])*np_weight.shape[0]
        # elif isinstance(layer, nn.Linear):
        #   total = (np_weight.shape[1])*np_weight.shape[0]
            total = np_weight.size
        # Print sparsity
            print('Sparsity of '+name+': '+str(zeros/total))
        if (isinstance(layer, nn.Linear) and name in exempt_layer):

        # Optional: Check the sparsity you achieve in each layer
        # Convert the weight of "layer" to numpy array
            np_weight = layer.weight.data.cpu().detach().numpy()
        # Count number of zeros
            zeros = np.count_nonzero(np_weight==0)
        # Count number of parameters
        # if isinstance(layer, nn.Conv2d):
        #   total = (np_weight.shape[3]*np_weight.shape[2]*np_weight.shape[1])*np_weight.shape[0]
        # elif isinstance(layer, nn.Linear):
        #   total = (np_weight.shape[1])*np_weight.shape[0]
            total = np_weight.size
            nonzeros = total-zeros
        # Print sparsity
            print('Sparsity of '+name+': '+str(nonzeros/total))

    print("Generating Video using", len(data), "different view points")
    # rgb_frames = []
    count = 0
    # if config.visualize_depth:
    #    depth_frames = []
    # if config.visualize_normals:
    #     normal_frames = []
    for ray in tqdm(data):
        # img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            img, _, _ = model.render_image(ray, data.h, data.w, chunks=config.chunks)
        # rgb_frames.append(img)
            imageio.imwrite(path.join(config.log_dir, f"image_{count}.jpeg"), img)
        count += 1
        #if config.visualize_depth:
        #    depth_frames.append(to8b(visualize_depth(dist, acc, data.near, data.far)))
        #if config.visualize_normals:
        #    normal_frames.append(to8b(visualize_normals(dist, acc)))

   # imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=30, quality=10)
   # if config.visualize_depth:
   #     imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=30, quality=10, codecs="hvec")
   # if config.visualize_normals:
    #    imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=30, quality=10, codecs="hvec")


if __name__ == "__main__":
    config = get_config()
    visualize(config)
