import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from dlkit.utils.config import load_config
from dlkit.deploy.infer import Inferencer


def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='image file or directory')
    parser.add_argument('--output-dir', default='runs/predict')
    parser.add_argument('--device', default=None)
    parser.add_argument('--palette', default=None, help='path to classes.json with colors')
    args = parser.parse_args()

    cfg = load_config(args.config)
    inferencer = Inferencer(cfg, args.checkpoint, device=args.device, palette=args.palette)

    if os.path.isdir(args.input):
        files = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    else:
        files = [args.input]

    for f in files:
        outputs = inferencer.predict_file(f, output_dir=args.output_dir)
        print('%s -> %s' % (f, outputs))


if __name__ == '__main__':
    main()
