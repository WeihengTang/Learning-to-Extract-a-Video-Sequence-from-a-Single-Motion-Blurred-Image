from __future__ import print_function
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
parser.add_argument('--cuda', action='store_true', help='use cuda')

args = parser.parse_args()

# Determine device for loading models
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

# load model
model1 = centerEsti()
model2 = F35_N8()
model3 = F26_N9()
model4 = F17_N9()
checkpoint = torch.load('models/center_v3.pth', map_location=device)
model1.load_state_dict(checkpoint['state_dict_G'], strict=False)
checkpoint = torch.load('models/F35_N8.pth', map_location=device)
model2.load_state_dict(checkpoint['state_dict_G'], strict=False)
checkpoint = torch.load('models/F26_N9_from_F35_N8.pth', map_location=device)
model3.load_state_dict(checkpoint['state_dict_G'], strict=False)
checkpoint = torch.load('models/F17_N9_from_F26_N9_from_F35_N8.pth', map_location=device)
model4.load_state_dict(checkpoint['state_dict_G'], strict=False)

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
output_data = output1[0]*255
utils.save_image(inputFile[:-4] + '-esti1' + inputFile[-4:], output_data)
output_data = output2[0]*255
utils.save_image(inputFile[:-4] + '-esti2' + inputFile[-4:], output_data)
output_data = output3[0]*255
utils.save_image(inputFile[:-4] + '-esti3' + inputFile[-4:], output_data)
output_data = output4[0]*255
utils.save_image(inputFile[:-4] + '-esti4' + inputFile[-4:], output_data)
output_data = output5[0]*255
utils.save_image(inputFile[:-4] + '-esti5' + inputFile[-4:], output_data)
output_data = output6[0]*255
utils.save_image(inputFile[:-4] + '-esti6' + inputFile[-4:], output_data)
output_data = output7[0]*255
utils.save_image(inputFile[:-4] + '-esti7' + inputFile[-4:], output_data)
