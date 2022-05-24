import argparse
import glob, os
import time
from pathlib import Path
from PIL import Image

import torch
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from network.Transformer import Transformer
from huggingface_hub import hf_hub_download

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    desc = "CartoonGAN CLI by soulteary"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='Shinkai', help='Shinkai / Hosoda / Miyazaki / Kon')
    parser.add_argument('--input', type=str, default='./images', help='images directory')
    parser.add_argument('--output', type=str, default='./result/', help='output path')
    parser.add_argument('--resize', type=int, default=0,
                        help='Do you need a program to adjust the image size?')
    parser.add_argument('--maxsize', type=int, default=0,
                        help='your desired image output size')
    """
    If you want to resize, you need to specify both --resize and --maxsize
    """
    return parser.parse_args()

def prepare_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


arg = parse_args()


enable_gpu = torch.cuda.is_available()

if enable_gpu:
    # If you have multiple cards,
    # you can assign to a specific card, eg: "cuda:0"("cuda") or "cuda:1"
    # Use the first card by default: "cuda"
    device = torch.device("cuda")
else:
    device = "cpu"

def get_model(style):
    # Makoto Shinkai
    if style == "Shinkai":
        MODEL_REPO_SHINKAI = "akiyamasho/AnimeBackgroundGAN-Shinkai"
        MODEL_FILE_SHINKAI = "shinkai_makoto.pth"
        model_hfhub = hf_hub_download(repo_id=MODEL_REPO_SHINKAI, filename=MODEL_FILE_SHINKAI)
    # Mamoru Hosoda
    elif style == "Hosoda":
        MODEL_REPO_HOSODA = "akiyamasho/AnimeBackgroundGAN-Hosoda"
        MODEL_FILE_HOSODA = "hosoda_mamoru.pth"
        model_hfhub = hf_hub_download(repo_id=MODEL_REPO_HOSODA, filename=MODEL_FILE_HOSODA)
    # Hayao Miyazaki
    elif style == "Miyazaki":
        MODEL_REPO_MIYAZAKI = "akiyamasho/AnimeBackgroundGAN-Miyazaki"
        MODEL_FILE_MIYAZAKI = "miyazaki_hayao.pth"
        model_hfhub = hf_hub_download(repo_id=MODEL_REPO_MIYAZAKI, filename=MODEL_FILE_MIYAZAKI)
    # Satoshi Kon
    elif style == "Kon":
        MODEL_REPO_KON = "akiyamasho/AnimeBackgroundGAN-Kon"
        MODEL_FILE_KON = "kon_satoshi.pth"
        model_hfhub = hf_hub_download(repo_id=MODEL_REPO_KON, filename=MODEL_FILE_KON)

    model = Transformer()
    model.load_state_dict(torch.load(model_hfhub, device))
    if enable_gpu:
        model = model.to(device)
    model.eval()
    return model

def inference(img, model):
    # load image
    input_image = img.convert("RGB")
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image

    if enable_gpu:
        logger.info(f"CUDA found. Using GPU.")
        # Allows to specify a card for calculation
        input_image = Variable(input_image).to(device)
    else:
        logger.info(f"CUDA not found. Using CPU.")
        input_image = Variable(input_image).float()

    # forward
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    return transforms.ToPILImage()(output_image)


prepare_dirs(arg.output)

model = get_model(arg.model)

enable_resize = False
max_dimensions = -1
if arg.maxsize > 0:
    max_dimensions = arg.maxsize
    if arg.resize :
        enable_resize = True

globPattern = arg.input + "/*.png"

for filePath in glob.glob(globPattern):
    basename = os.path.basename(filePath)
    with Image.open(filePath) as img:
        if(enable_resize):
            img.thumbnail((max_dimensions, max_dimensions), Image.Resampling.LANCZOS)

        start_time = time.time()
        inference(img, model).save(arg.output + "/" + basename, "PNG")
        print("--- %s seconds ---" % (time.time() - start_time))
