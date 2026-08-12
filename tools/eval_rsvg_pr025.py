import runpy
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mmdet.evaluation.metrics.refexp_metric


if len(sys.argv) != 4:
    raise SystemExit('Usage: eval_rsvg_pr025.py <config> <checkpoint> <work_dir>')

config_path, checkpoint_path, work_dir = sys.argv[1:]
sys.argv = [
    'tools/test.py', config_path, checkpoint_path, '--cfg-options',
    'test_evaluator.iou_thrs=[0.25,0.5]', f'work_dir={work_dir}'
]
runpy.run_path('tools/test.py', run_name='__main__')
