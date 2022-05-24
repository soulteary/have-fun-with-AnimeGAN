import argparse
import towhee

def parse_args():
    desc = "AnimeGAN CLI by soulteary"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='Shinkai', help='origin / facepaintv2 / hayao / paprika / shinkai')
    parser.add_argument('--device', type=str, default='cpu', help='cpu / cuda')
    parser.add_argument('--input', type=str, default='./images', help='images directory')
    parser.add_argument('--output', type=str, default='./result/', help='output path')
    """
    If you want to resize, you need to specify both --resize and --maxsize
    """
    return parser.parse_args()

arg = parse_args()
towhee.glob['path'](arg.input + "/*.png") \
    .set_parallel(5) \
    .image_decode['path', 'img']() \
    .img2img_translation.animegan['img', 'new_img'](model_name = arg.model, device=arg.device) \
    .save_image[('new_img', 'path'), 'new_path'](dir=arg.output + "/", format="png") \
    .to_list()