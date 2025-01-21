import torch
import torchvision.utils as tvu
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r, alpha):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.lora_up = nn.Linear(original_layer.in_features, r, bias=False)
        self.lora_down = nn.Linear(r, original_layer.out_features, bias=False)
        self.scaling = self.alpha / self.r

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down.weight)

    def forward(self, x):
        return self.original_layer(x) + self.lora_down(self.lora_up(x)) * self.scaling

def add_lora_adapters(unet, r, alpha):
    modules_to_modify = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_modify.append((name, module))
    
    for name, module in modules_to_modify:
        parent_module = unet
        for part in name.split('.')[:-1]:
            parent_module = getattr(parent_module, part)
        last_part = name.split('.')[-1]
        setattr(parent_module, last_part, LoRALayer(module, r, alpha))
    
    return unet