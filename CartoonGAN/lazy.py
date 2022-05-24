import argparse
import towhee

def parse_args():
    desc = "CartoonGAN CLI by soulteary"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='Shinkai', help='Shinkai / Hosoda / Miyazaki / Kon')
    parser.add_argument('--input', type=str, default='./images', help='images directory')
    parser.add_argument('--output', type=str, default='./result/', help='output path')
    """
    If you want to resize, you need to specify both --resize and --maxsize
    """
    return parser.parse_args()

arg = parse_args()
towhee.glob(arg.input + "/*.png") \
    .set_parallel(5) \
    .image_decode() \
    .img2img_translation.cartoongan(model_name = arg.model) \
    .save_image(dir=arg.output + "/")