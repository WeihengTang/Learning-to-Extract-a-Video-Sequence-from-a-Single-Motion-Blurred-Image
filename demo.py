from __future__ import print_function
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import utils
from model import centerEsti
from model import F26_N9
from model import F17_N9
from model import F35_N8

# Training settings
parser = argparse.ArgumentParser(description='parser for video prediction')
parser.add_argument('--input', type=str, required=True, help='input image')
parser.add_argument('--output_dir', type=str, default='results', help='output directory')
parser.add_argument('--cuda', action='store_true', help='use cuda')

args = parser.parse_args()

# Determine device for loading models
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

def _load_torch(path, device):
    """torch.load wrapper compatible with both PyTorch 1.x and 2.x."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def load_checkpoint(path, model, device):
    """Load checkpoint, filtering out InstanceNorm running stats for PyTorch 0.4+ compatibility."""
    checkpoint = _load_torch(path, device)
    state_dict = checkpoint['state_dict_G']
    # Filter out running_mean/running_var (not used with track_running_stats=False)
    filtered = {k: v for k, v in state_dict.items()
                if 'running_mean' not in k and 'running_var' not in k and 'num_batches' not in k}
    model.load_state_dict(filtered, strict=False)

# load model
model1 = centerEsti()
model2 = F35_N8()
model3 = F26_N9()
model4 = F17_N9()
load_checkpoint('models/center_v3.pth', model1, device)
load_checkpoint('models/F35_N8.pth', model2, device)
load_checkpoint('models/F26_N9_from_F35_N8.pth', model3, device)
load_checkpoint('models/F17_N9_from_F26_N9_from_F35_N8.pth', model4, device)

if args.cuda:
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()

model1.eval()
model2.eval()
model3.eval()
model4.eval()

inputFile = args.input
input = utils.load_image(inputFile)
width, height= input.size
input = input.crop((0,0, width-width%20, height-height%20))
input_transform = transforms.Compose([
    transforms.ToTensor(),
])
input = input_transform(input)
input = input.unsqueeze(0)
if args.cuda:
    input = input.cuda()

# Use torch.no_grad() instead of deprecated volatile=True
with torch.no_grad():
    output4 = model1(input)
    output3_5 = model2(input, output4)
    output2_6 = model3(input, output3_5[0], output4, output3_5[1])
    output1_7 = model4(input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])
if args.cuda:
    output1 = output1_7[0].cpu()
    output2 = output2_6[0].cpu()
    output3 = output3_5[0].cpu()
    output4 = output4.cpu()
    output5 = output3_5[1].cpu()
    output6 = output2_6[1].cpu()
    output7 = output1_7[1].cpu()
else:
    output1 = output1_7[0]
    output2 = output2_6[0]
    output3 = output3_5[0]
    output4 = output4
    output5 = output3_5[1]
    output6 = output2_6[1]
    output7 = output1_7[1]
baseName, ext = os.path.splitext(os.path.basename(inputFile))
seq_dir = os.path.join(args.output_dir, baseName)
os.makedirs(seq_dir, exist_ok=True)
for i, out in enumerate([output1, output2, output3, output4, output5, output6, output7]):
    output_data = out[0] * 255
    save_path = os.path.join(seq_dir, f'frame_{i:02d}.png')
    utils.save_image(save_path, output_data)
