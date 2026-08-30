import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from dlkit.utils.config import load_config
from dlkit.engine.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on a split')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default=None, help='save metrics to json')
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = evaluate(cfg, args.checkpoint, split=args.split, device=args.device, output_json=args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
