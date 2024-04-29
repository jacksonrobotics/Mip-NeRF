import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import torch
import torchvision
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.utils.prune as prune

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        model.eval()  # Set model to evaluation mode after loading
        print("Model loaded successfully with adjusted keys.")
    except RuntimeError as e:
        print(f"Failed to load model: {e}")

def prune_by_percentage(layer, q=50.0):
    """
    Pruning the weight paramters by threshold.
    :param q: pruning percentile. 'q' percent of the least
    significant weight parameters will be pruned.
    """
    # Convert the weight of "layer" to numpy array

    # weight = layer.weight.data.cpu().detach().numpy()
    weight = layer.weight.data

    # Compute the q-th percentile of the abs of the converted array
    perc = torch.percentile(np.abs(weight), q)

    # Generate a binary mask same shape as weight to decide which element to prune
    mask = torch.where(np.abs(weight) < perc, 0, 1)

    # Convert mask to torch tensor and put on GPU

    tensor_mask = torch.tensor(mask).to(device)

    # Multiply the weight by mask to perform pruning
    # masked_layer = torch.mul(new_layer.weight.data.to(device), tensor_mask)
    tensor_weight = torch.tensor(weight).to(device)
    masked_layer = torch.mul(tensor_weight, tensor_mask).to(device)
    # print(masked_layer.is_cuda)

    layer.weight.data = masked_layer

    return layer


def train_model(config):
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))

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
        Nbits=8,  # Add Nbits parameter for quantization
        symmetric=False  # Add symmetric parameter for quantizationNbits=None,  # Add Nbits parameter for quantization
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        # model.load_state_dict(torch.load(model_save_path))
        load_pruned_state_dict(model, torch.load(model_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()

    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter()
    q = 0
    exempt_layer = ['rgb_net0.0', 'rgb_net1.0', 'final_density.0', 'final_rgb.0']
    # prune_val = True
    current_prune_level = 0

    weight_mask = {}
    for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear)) and name not in exempt_layer:
                weight_mask[name] = torch.tensor(np.where(np.abs(module.weight.data.cpu().detach().numpy()) != 0, 1, 0)).to(device)

    n = True
    for step in tqdm(range(0, config.max_steps)):
        rays, pixels = next(data)
        comp_rgb, _, _ = model(rays)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        current_progress = (step - 0) / config.max_steps

        # pruning
        # if step < 2500:
        #     if step%250 == 0:
        #         current_prune_level += 0.01
        #         for name,layer in model.named_modules():
        #             if (isinstance(layer, nn.Linear) and name not in exempt_layer):
        #                 prune.ln_structured(layer, name='weight', amount=current_prune_level, n=1, dim=0)
        #                 print(f'Pruned coarse {name} to {current_prune_level * 100:.2f}% at {current_progress * 100:.1f}% of training')
        #                 # # change q value
        #                 # q += 1
        #                 # prune_by_percentage(layer, q=q)

        

        while n:
        # if step%1000 == 0 and step<5000:
                            
                            current_prune_level += 0.01
                            for name,layer in model.named_modules():
                                if (isinstance(layer, nn.Linear) and name not in exempt_layer):
                                    prune.ln_structured(layer, name='weight', amount=0.1, n=1, dim=0)
                                    print(f'Pruned coarse {name} to {0.1 * 100:.2f}% at {current_progress * 100:.1f}% of training')

                            # for name,layer in model.named_modules():
                            #     if (isinstance(layer, nn.Linear) and name not in exempt_layer):
                            #         np_weight = layer.weight.data.cpu().detach().numpy()
                            #         zeros = np.count_nonzero(np_weight==0)
                            #         total = np_weight.size
                            #         print('Sparsity of '+name+': '+str(zeros/total))

                            #     if (isinstance(layer, nn.Linear) and name in exempt_layer):
                            #         np_weight = layer.weight.data.cpu().detach().numpy()
                            #         zeros = np.count_nonzero(np_weight==0)
                            #         total = np_weight.size
                            #         nonzeros = total-zeros
                            #         print('Sparsity of '+name+': '+str(nonzeros/total))
                            n = False
        if step%1000 == 0 and step<5000:
             for name,layer in model.named_modules():
                                if (isinstance(layer, nn.Linear) and name not in exempt_layer):
                                    np_weight = layer.weight.data.cpu().detach().numpy()
                                    zeros = np.count_nonzero(np_weight==0)
                                    total = np_weight.size
                                    print('Sparsity of '+name+': '+str(zeros/total))

                                if (isinstance(layer, nn.Linear) and name in exempt_layer):
                                    np_weight = layer.weight.data.cpu().detach().numpy()
                                    zeros = np.count_nonzero(np_weight==0)
                                    total = np_weight.size
                                    nonzeros = total-zeros
                                    print('Sparsity of '+name+': '+str(nonzeros/total))

        for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear)) and name not in exempt_layer:
                    module.weight.data = module.weight.data * weight_mask[name]

        #Check the weights anyway
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear)) and name not in exempt_layer:
                non_zero_weights = torch.count_nonzero(module.weight).item()
                logger.add_scalar(f'Non-zero weights/coarse_{name}', non_zero_weights, step)

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)

        if step % config.save_every == 0:
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().cpu().numpy()
                logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
                logger.add_scalar('eval/fine_psnr', float(psnr[-1]), global_step=step)
                logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)), global_step=step)
                # logger.add_image('eval/image_target', cast_to_image(), global_step=step)

            torch.save(model.state_dict(), model_save_path+f"_{step}")
            torch.save(optimizer.state_dict(), optimizer_save_path+f"_{step}")

    torch.save(model.state_dict(), model_save_path+f"_{step}")
    torch.save(optimizer.state_dict(), optimizer_save_path+f"_{step}")


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])

def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

if __name__ == "__main__":
    config = get_config()
    train_model(config)
