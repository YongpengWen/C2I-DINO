from argparse import ArgumentParser
import os

from mmdet.apis import DetInferencer


def parse_args():
    parser = ArgumentParser(description='Run Grounding DINO on one OPT image.')
    parser.add_argument(
        '--config',
        default='configs/opt_vg/grounding_dino_swin-t_finetune_opt_vg.py')
    parser.add_argument(
        '--weights',
        required=True,
        help='Path or URL of a Grounding DINO checkpoint.')
    parser.add_argument(
        '--image',
        default=os.path.join(
            os.getenv('OPT_DATA_ROOT', '/root/autodl-tmp/opt-rsvg'),
            'Image', '000000.jpg'))
    parser.add_argument(
        '--text',
        default='airplane . ship . vehicle . bridge . storage tank .')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', default='outputs/opt_vg_demo')
    parser.add_argument('--score-thr', type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = DetInferencer(
        model=args.config,
        weights=args.weights,
        device=args.device,
        palette='random')
    inferencer(
        args.image,
        texts=args.text,
        custom_entities=True,
        pred_score_thr=args.score_thr,
        out_dir=args.out_dir)
    print(f'Done. Results saved to {args.out_dir}')


if __name__ == '__main__':
    main()
