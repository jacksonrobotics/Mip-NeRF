from model import MipNeRF
from config import get_config
import torch

# regular Mip-Ner: 2.338 mb

# def visualize(config):
#     model = MipNeRF(
#             use_viewdirs=config.use_viewdirs,
#             randomized=config.randomized,
#             ray_shape=config.ray_shape,
#             white_bkgd=config.white_bkgd,
#             num_levels=config.num_levels,
#             num_samples=config.num_samples,
#             hidden=config.hidden,
#             density_noise=config.density_noise,
#             density_bias=config.density_bias,
#             rgb_padding=config.rgb_padding,
#             resample_padding=config.resample_padding,
#             min_deg=config.min_deg,
#             max_deg=config.max_deg,
#             viewdirs_min_deg=config.viewdirs_min_deg,
#             viewdirs_max_deg=config.viewdirs_max_deg,
#             device=config.device,
#         )
#     param_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#     buffer_size = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()

#     size_all_mb = (param_size + buffer_size) / 1024**2
#     print('model size: {:.3f}MB'.format(size_all_mb))

checkpoint = torch.load("log/8_quant_lego_5k/model.pt_4999")
checkpoint = torch.load("log/Success_ficus/model.pt")
size_model = 0
for param in checkpoint.values():
    if param.is_floating_point():
        size_model += param.numel() * torch.finfo(param.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.dtype).bits
print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

# if __name__ == "__main__":
#     config = get_config()
#     visualize(config)