import argparse
import towhee

def parse_args():
    desc = "AnimeGAN CLI by soulteary"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='Shinkai', help='origin / facepaintv2 / hayao / paprika / shinkai')
    parser.add_argument('--device', type=str, default='cpu', help='cpu / cuda')
    parser.add_argument('--input', type=str, default='./video.mp4', help='video filename')
    parser.add_argument('--output', type=str, default='./result/', help='output path')
    parser.add_argument('--rate', type=int, default=15, help='video rate')
    """
    If you want to resize, you need to specify both --resize and --maxsize
    """
    return parser.parse_args()


arg = parse_args()
towhee.read_video(arg.input) \
    .set_parallel(5) \
    .image_resize(fx=0.2, fy=0.2) \
    .img2img_translation.animegan(model_name = arg.model, device=arg.device) \
    .to_video(arg.output + '/result.mp4', 'x264', arg.rate)