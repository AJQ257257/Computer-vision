import argparse
import time

import PIL
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', default='Set5/head.png', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_36.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.to(device)
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME).convert('RGB')

image_width = (image.width // UPSCALE_FACTOR) * UPSCALE_FACTOR
image_height = (image.height // UPSCALE_FACTOR) * UPSCALE_FACTOR
image = image.resize((image_width, image_height), resample=PIL.Image.BICUBIC)
image = image.resize((image.width // UPSCALE_FACTOR, image.height // UPSCALE_FACTOR), resample=PIL.Image.BICUBIC)
image = image.resize((image.width * UPSCALE_FACTOR, image.height * UPSCALE_FACTOR), resample=PIL.Image.BICUBIC)
image.save(IMAGE_NAME.replace('.', '_bicubic_x{}.'.format(UPSCALE_FACTOR)))

image = Image.open(IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.to(device)


out = model(image)

out_img = ToPILImage()(out[0].data.cpu())
out_img.save(IMAGE_NAME.replace('.', '_srcnn_x{}.'.format(UPSCALE_FACTOR)))
# out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
