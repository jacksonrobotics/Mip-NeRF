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
import copy
from skimage.metrics import structural_similarity as ssim
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter

def train_model(config):
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    eval_data = None
    # if config.do_eval:
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
        model.load_state_dict(torch.load(model_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))
    
    for name, param in model.named_parameters():
        print(name, param.dtype)
    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = NeRFLoss(config.coarse_weight_decay)

    model.train()

    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter()

    for step in tqdm(range(0, config.max_steps)):
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        rays, pixels = next(data)
        comp_rgb, _, _ = model(rays)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)

        if step % config.save_every == 0:

            # img, _, _ = model.render_image(rays, data.h, data.w, chunks=config.chunks)
            # train_img = cast_to_image(comp_rgb[..., :3])
            # logger.add_scalar('train/image_target', target_img, global_step=step)
            # logger.add_scalar('train/image', train_img, global_step=step)
            # logger.add_scalar('ssim', ssim(target_img, train_img, full=True), global_step=step)
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().cpu().numpy()
                logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
                logger.add_scalar('eval/fine_psnr', float(psnr[-1]), global_step=step)
                logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)), global_step=step)
                # logger.add_image('eval/image_target', cast_to_image(eval_data), global_step=step)



            torch.save(model.state_dict(), model_save_path+f"_{step}")
            torch.save(optimizer.state_dict(), optimizer_save_path+f"_{step}")

    torch.save(model.state_dict(), model_save_path+f"_{step}")
    torch.save(optimizer.state_dict(), optimizer_save_path+f"_{step}")

    for name, module in model.named_modules():
            for param_name, param_value in module.named_parameters():
                if 'weight' in param_name:  # Ensure only weights are logged
                    logger.add_histogram(f"Quantized Mip-NeRF {name}/{param_name}", param_value, step)
                    logger.flush()


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
