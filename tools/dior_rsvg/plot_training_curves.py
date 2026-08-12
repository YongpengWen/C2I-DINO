import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot MMEngine scalar logs for DIOR-RSVG training.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/dior_rsvg',
        help='Training work directory.')
    parser.add_argument(
        '--scalars',
        default=None,
        help='Path to vis_data/scalars.json. If omitted, use the latest run '
        'under work-dir.')
    parser.add_argument(
        '--out',
        default=None,
        help='Output png path. Defaults to WORK_DIR/training_curves.png.')
    return parser.parse_args()


def find_latest_scalars(work_dir):
    candidates = sorted(
        Path(work_dir).glob('*/vis_data/scalars.json'),
        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'No scalars.json found under {work_dir}')
    return candidates[-1]


def load_scalars(path):
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_series(ax, rows, keys, title, ylabel):
    plotted = False
    for key in keys:
        pts = [(row.get('step'), row.get(key)) for row in rows if key in row]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=key)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            'No data yet',
            ha='center',
            va='center',
            transform=ax.transAxes)


def metric_keys(rows):
    keys = set()
    for row in rows:
        for key in row:
            if ('Pr@' in key or 'meanIoU' in key or 'cumIoU' in key):
                keys.add(key)
    return sorted(keys)


def main():
    args = parse_args()
    scalars_path = Path(args.scalars) if args.scalars else find_latest_scalars(
        args.work_dir)
    out_path = Path(args.out) if args.out else Path(
        args.work_dir) / 'training_curves.png'
    rows = load_scalars(scalars_path)
    if not rows:
        raise RuntimeError(f'No scalar rows in {scalars_path}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_series(axes[0][0], rows, ['loss'], 'Total Loss', 'loss')
    plot_series(axes[0][1], rows, ['loss_cls', 'loss_bbox', 'loss_iou'],
                'Main Loss Components', 'loss')
    plot_series(axes[1][0], rows, ['lr', 'base_lr'], 'Learning Rate', 'lr')
    plot_series(axes[1][1], rows, metric_keys(rows), 'Validation Metrics',
                'metric')

    fig.suptitle(scalars_path.parent.parent.name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
