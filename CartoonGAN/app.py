import torch
import gradio as gr
import numpy as np
import torchvision.transforms as transforms


from torch.autograd import Variable
from network.Transformer import Transformer
from huggingface_hub import hf_hub_download

from PIL import Image

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants

MAX_DIMENSION = 1280
MODEL_PATH = "models"
COLOUR_MODEL = "RGB"

STYLE_SHINKAI = "Makoto Shinkai"
STYLE_HOSODA = "Mamoru Hosoda"
STYLE_MIYAZAKI = "Hayao Miyazaki"
STYLE_KON = "Satoshi Kon"
DEFAULT_STYLE = STYLE_SHINKAI
STYLE_CHOICE_LIST = [STYLE_SHINKAI, STYLE_HOSODA, STYLE_MIYAZAKI, STYLE_KON]

MODEL_REPO_SHINKAI = "akiyamasho/AnimeBackgroundGAN-Shinkai"
MODEL_FILE_SHINKAI = "shinkai_makoto.pth"

MODEL_REPO_HOSODA = "akiyamasho/AnimeBackgroundGAN-Hosoda"
MODEL_FILE_HOSODA = "hosoda_mamoru.pth"

MODEL_REPO_MIYAZAKI = "akiyamasho/AnimeBackgroundGAN-Miyazaki"
MODEL_FILE_MIYAZAKI = "miyazaki_hayao.pth"

MODEL_REPO_KON = "akiyamasho/AnimeBackgroundGAN-Kon"
MODEL_FILE_KON = "kon_satoshi.pth"

# Model Initalisation
shinkai_model_hfhub = hf_hub_download(repo_id=MODEL_REPO_SHINKAI, filename=MODEL_FILE_SHINKAI)
hosoda_model_hfhub = hf_hub_download(repo_id=MODEL_REPO_HOSODA, filename=MODEL_FILE_HOSODA)
miyazaki_model_hfhub = hf_hub_download(repo_id=MODEL_REPO_MIYAZAKI, filename=MODEL_FILE_MIYAZAKI)
kon_model_hfhub = hf_hub_download(repo_id=MODEL_REPO_KON, filename=MODEL_FILE_KON)

shinkai_model = Transformer()
hosoda_model = Transformer()
miyazaki_model = Transformer()
kon_model = Transformer()

enable_gpu = torch.cuda.is_available()

if enable_gpu:
    # If you have multiple cards,
    # you can assign to a specific card, eg: "cuda:0"("cuda") or "cuda:1"
    # Use the first card by default: "cuda"
    device = torch.device("cuda")
else:
    device = "cpu"

shinkai_model.load_state_dict(
    torch.load(shinkai_model_hfhub, device)
)
hosoda_model.load_state_dict(
    torch.load(hosoda_model_hfhub, device)
)
miyazaki_model.load_state_dict(
    torch.load(miyazaki_model_hfhub, device)
)
kon_model.load_state_dict(
    torch.load(kon_model_hfhub, device)
)

if enable_gpu:
    shinkai_model = shinkai_model.to(device)
    hosoda_model = hosoda_model.to(device)
    miyazaki_model = miyazaki_model.to(device)
    kon_model = kon_model.to(device)

shinkai_model.eval()
hosoda_model.eval()
miyazaki_model.eval()
kon_model.eval()


# Functions

def get_model(style):
    if style == STYLE_SHINKAI:
        return shinkai_model
    elif style == STYLE_HOSODA:
        return hosoda_model
    elif style == STYLE_MIYAZAKI:
        return miyazaki_model
    elif style == STYLE_KON:
        return kon_model
    else:
        logger.warning(
            f"Style {style} not found. Defaulting to Makoto Shinkai"
        )
        return shinkai_model


def adjust_image_for_model(img):
    logger.info(f"Image Height: {img.height}, Image Width: {img.width}")
    if img.height > MAX_DIMENSION or img.width > MAX_DIMENSION:
        logger.info(f"Dimensions too large. Resizing to {MAX_DIMENSION}px.")
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)

    return img


def inference(img, style):
    img = adjust_image_for_model(img)

    # load image
    input_image = img.convert(COLOUR_MODEL)
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
    model = get_model(style)
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    return transforms.ToPILImage()(output_image)


# Gradio setup

title = "Anime Background GAN"
description = "Gradio Demo for CartoonGAN by Chen Et. Al. Models are Shinkai Makoto, Hosoda Mamoru, Kon Satoshi, and Miyazaki Hayao."
article = "<p style='text-align: center'><a href='http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf' target='_blank'>CartoonGAN Whitepaper from Chen et.al</a></p><p style='text-align: center'><a href='https://github.com/venture-anime/cartoongan-pytorch' target='_blank'>Github Repo</a></p><p style='text-align: center'><a href='https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch' target='_blank'>Original Implementation from Yijunmaverick</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akiyamasho' alt='visitor badge'></center></p>"

examples = [
    ["examples/garden_in.jpg", STYLE_SHINKAI],
    ["examples/library_in.jpg", STYLE_KON],
]


gr.Interface(
    fn=inference,
    inputs=[
        gr.inputs.Image(
            type="pil",
            label="Input Photo (less than 1280px on both width and height)",
        ),
        gr.inputs.Dropdown(
            STYLE_CHOICE_LIST,
            type="value",
            default=DEFAULT_STYLE,
            label="Style",
        ),
    ],
    outputs=gr.outputs.Image(
        type="pil",
        label="Output Image",
    ),
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging="never",
).launch(enable_queue=True)
