import os
import math
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torchvision.utils as tvu
import torch.nn as nn
from pathlib import Path
from models.diffusion import Model
from diffusers import DDPMPipeline
from functions.process_data import *
from functions.file_utils import *
from functions.add_lora import *

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output.sample)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def image_editing_sample(self):
        print("Loading model")
        pipeline = DDPMPipeline.from_pretrained('./pretrained_models/ddpm-church-256')
        model = pipeline.unet
        model = add_lora_adapters(model, 4, 4)
        # model.load_state_dict(torch.load('./results/071303/checkpoint-14000/unet.ckpt')) # Si
        # model.load_state_dict(torch.load('./results/072802/checkpoint-16000/unet.ckpt')) # Si+STO
        # model.load_state_dict(torch.load('./results/GaN-0822/checkpoint-12000/unet.ckpt')) # GaN
        model.load_state_dict(torch.load('./results/053001/checkpoint-5000/unet.ckpt')) # MoS2
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0

        n = 1
        model.eval()
        print("Start sampling")
        
        base_dir = './data/MoS2/cut-s128/'
        image_files = find_all_files(base_dir, get_extensions_by_type('img'))
        
        image_files.sort(key=lambda x: (x[:3], int(float(x[x.rfind('_')+1:x.rfind('.')]))))

        print('load dataset successfully. {} images are loaded from {}.\n.'.format(len(image_files), base_dir))
        with torch.no_grad():
            result_folder = os.path.join(self.args.image_folder, 'results')
            os.makedirs(result_folder, exist_ok=True)
            for path in image_files:
                print(os.path.basename(path))
                mask = cv2.cvtColor(cv2.imread('black_mask.png'), cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                mask = torch.FloatTensor((mask - np.min(mask)) / np.ptp(mask) if np.ptp(mask) else mask / np.max(mask) if np.max(mask) else mask)
                img = torch.FloatTensor((img - np.min(img)) / np.ptp(img))
                mask = mask.to(self.config.device).permute(2, 0, 1)
                img = img.to(self.config.device)
                img = img.unsqueeze(dim=0)
                img = img.repeat(n, 1, 1, 1)
                x0 = img.permute(0, 3, 1, 2)

                tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input.png'))
                x0 = (x0 - 0.5) * 2.

                for it in range(self.args.sample_step):
                    e = torch.randn_like(x0)
                    total_noise_levels = self.args.t
                    a = (1 - self.betas).cumprod(dim=0)

                    x = x0
                    tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))

                    with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                        for i in reversed(range(total_noise_levels)):
                            t = (torch.ones(n) * i).to(self.device)
                            x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                            logvar=self.logvar,
                                                                            betas=self.betas)
                            x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                            x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                            progress_bar.update(1)

                    x0[:, (mask != 1.)] = x[:, (mask != 1.)]
                    tvu.save_image((x + 1) * 0.5, os.path.join(result_folder,
                                                                f'{Path(path).stem}.png'))
