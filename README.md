# 遥感视觉指代表达：Grounding DINO 实验代码

这是可直接发布到 GitHub 的代码整理版。训练日志、checkpoint、实验可视化和完整数据集均未包含；预训练模型已放在 `pretrained/`，也可改为从 Git LFS、Release 或模型仓库下载。

本仓库是一个基于 [Grounding DINO](https://github.com/open-mmlab/mmdetection) 和 MMDetection 的遥感视觉指代表达（Referring Expression Grounding, REG）实验项目。代码面向 RSVG、DIOR-RSVG、OPT-RSVG 和 FLIR 等遥感/红外数据集，支持根据自然语言描述在图像中定位目标，并以 `Pr@IoU`、mean IoU 等指标进行评估。

仓库包含 Grounding DINO 的实验配置、类别后缀（class suffix）和空间对比学习（spatial contrast）等改动。根目录的 `dual_grounding_dino_enhance_share_w.py` 还提供了一个红外（IR）+可见光（VI）双分支旋转框 Grounding DINO 原型，包含多尺度特征融合和 WFAC 特征增强逻辑；该文件属于实验性代码，使用前请先完成对应模块注册和配置接入。

## 方法概览

- 以 Swin-T backbone 的 Grounding DINO 为基础进行文本-视觉跨模态对齐。
- 使用可学习的类别特定文本后缀（class-specific suffix）增强类别语义表示。
- 使用空间对比损失抑制相邻/易混淆候选框的错误匹配。
- 针对 RSVG、DIOR-RSVG、OPT-RSVG 和 FLIR 提供独立的数据处理与评估配置。
- 双模态原型 `SharedDualRotatedGroundingDINOEnhanceAdd` 支持 IR/VI 两路 backbone，并在多尺度特征上进行融合。

## 目录结构

```text
.
├── README.md
├── dual_grounding_dino_enhance_share_w.py   # 双模态旋转框模型原型
├── configs/                                  # RSVG、DIOR-RSVG、OPT-RSVG、FLIR 配置
├── mmdet/                                    # MMDetection 与本项目自定义模块
├── tools/                                    # 训练、测试、数据转换和评估脚本
├── scripts/                                  # 常用训练脚本
├── docs/                                    # 移植、许可证和发布检查说明
│   ├── PORTING.md
│   ├── LICENSES.md
│   └── RELEASE_CHECKLIST.md
├── pretrained/                              # Grounding DINO 和 BERT 预训练权重
└── .gitignore / .gitattributes              # 排除数据集、日志和训练输出；大权重使用 Git LFS
```

## 环境

当前实验环境记录如下（来自训练日志）：

| 组件 | 版本 |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.0.1+cu117 |
| TorchVision | 0.15.2+cu117 |
| CUDA Runtime | 11.7（机器 NVCC 12.1） |
| MMCV | 2.0.0 |
| MMEngine | 0.10.4 |
| MMDetection | 3.3.0 |
| Transformers | 4.46.3 |
| GPU | NVIDIA GeForce RTX 4090（单卡） |

建议使用 Python 3.8、CUDA 11.7 对应的 PyTorch 环境。不同 CUDA/显卡环境可能需要重新安装匹配版本的 `mmcv`。

```bash
conda create -n mmdet python=3.8 -y
conda activate mmdet

# 按照 PyTorch 官方页面安装与 CUDA 匹配的 torch/torchvision
# 然后安装仓库依赖
cd .
pip install -r requirements/runtime.txt
pip install -v -e .
```

验证安装：

```bash
python - <<'PY'
import torch, mmcv, mmengine, mmdet, transformers
print('torch:', torch.__version__)
print('mmcv:', mmcv.__version__)
print('mmengine:', mmengine.__version__)
print('mmdet:', mmdet.__version__)
print('transformers:', transformers.__version__)
print('cuda:', torch.cuda.is_available())
PY
```

## 数据准备

数据集及其许可证请以各自官方发布页面为准。请自行下载原始图像和标注，不要在未经许可的情况下将完整数据集上传到 GitHub。

当前配置期望的数据组织形式如下：

```text
datasets/
├── RSVG/rsvg/
│   ├── mdetr_annotations*/
│   └── odvg_ann*/
├── DIOR-RSVG/
│   ├── mdetr_annotations*/
│   └── odvg_ann*/
├── opt-rsvg/
│   ├── mdetr_annotations*/
│   ├── odvg_ann*/
│   └── split/
└── flir/
    ├── mdetr_annotations/
    ├── odvg_ann/
    └── image_data/flir/ir/
```

配置文件中部分实验配置仍保留原训练机路径（例如 `/root/autodl-tmp/...`）。迁移到其他机器时，请修改对应配置中的 `data_root`、`mdetr_ann_root`、`image_prefix`、`lang_model_name`、`load_from` 和 `work_dir`。建议使用环境变量或相对路径；不要把本地数据集提交到 GitHub。

## 预训练模型

默认实验使用：

- Grounding DINO Swin-T 预训练权重：`pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth`
- BERT：`pretrained/bert-base-uncased/`

权重文件体积较大。发布到 GitHub 前请安装并初始化 Git LFS（`git lfs install`），再执行 `git lfs track "pretrained/*.pth" "pretrained/**/*.safetensors"`；或者将权重放到 GitHub Release/Hugging Face，并在配置中更新 `load_from` 与 `lang_model_name`。本目录中的权重仅供本地发布准备使用。

## 训练与测试

在仓库根目录执行。下面是当前项目中用于主要实验的配置：

```bash
cd .
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# RSVG：class suffix + spatial contrast
python tools/train.py \
  configs/rsvg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_rsvg_1024_12e.py \
  --work-dir work_dirs/rsvg

# DIOR-RSVG
python tools/train.py \
  configs/dior_rsvg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_dior_rsvg_official_split_resize640_12e.py \
  --work-dir work_dirs/dior_rsvg

# OPT-RSVG
python tools/train.py \
  configs/opt_vg/grounding_dino_swin-t_class_suffix_spatial_contrast_finetune_opt_vg_official_split_12e.py \
  --work-dir work_dirs/opt_rsvg

# 测试：将路径替换为实际生成的 best checkpoint
python tools/test.py <config.py> <checkpoint.pth> --work-dir <test_output_dir>
```

也可以运行批量脚本（脚本中的 Python、数据集和输出目录均为当前机器路径，迁移前需要修改）：

```bash
bash scripts/run_paper_rerun_three_datasets.sh
```

单卡显存不足时，可通过配置覆盖 batch size、`num_workers`、输入尺寸或启用 AMP；修改后请记录完整配置和随机种子，便于复现。

## 已记录结果

以下数字来自仓库现有日志，仅作为当前环境下的参考结果；正式发布时请补充 checkpoint 下载地址、完整测试日志和精确实验设置。

| 数据集/实验 | 测试指标 |
|---|---:|
| RSVG（class suffix + spatial contrast，1024，12 epochs） | `Pr@0.25 = 0.6781`，`Pr@0.5 = 0.6333`，`meanIoU = 0.5241`，`cumIoU = 0.5006` |
| RSVG 验证集（同实验） | 最佳 `Pr@0.5 = 0.6420`（epoch 12） |
| DIOR-RSVG（640，训练日志） | 最佳已记录验证 `Pr@0.5 = 0.8509`（epoch 4） |
| OPT-RSVG / FLIR | 当前 README 未填入结果，请以对应测试日志为准 |

## 公开到 GitHub 前的清理清单

建议只提交源代码、配置、脚本、少量示例和文档：

1. 删除或忽略 `datasets/`、`pretrained/`、`work_dirs/`、`paper_rerun_logs/` 中的大文件和本机运行产物。
2. 清理配置中的 `/root/autodl-tmp`、`/root/miniconda3` 等绝对路径。
3. 检查日志、checkpoint、图片和标注中是否包含个人信息、服务器信息或未公开数据。
4. 明确 MMDetection 原始代码、预训练模型和各数据集的上游许可证。
5. 为本项目新增代码选择合适的许可证；不要默认将上游 Apache-2.0 许可证扩展为本项目全部内容。

## 引用

如果本代码或配置对你的研究有帮助，请同时引用 Grounding DINO、MMDetection 以及你实际使用的数据集论文。项目论文信息发布后，可将下面占位内容替换为正式 BibTeX：

```bibtex
@misc{your_project_2026,
  title  = {Your Project Title},
  author = {Your Name},
  year   = {2026},
  note   = {遥感视觉指代表达实验代码}
}
```

## 免责声明

本仓库主要用于科研复现和方法验证。由于数据版本、预训练权重、CUDA/PyTorch 版本、显存和随机性设置的差异，运行结果可能与日志中的数字不同。使用数据和模型前请遵守其原始许可证及所在地区的相关法律法规。
