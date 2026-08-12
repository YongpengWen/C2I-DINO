# 移植到其他机器

发布包不包含数据集和训练输出。使用前请自行下载并按根目录 README 中的结构放置数据。

建议设置以下环境变量或修改对应配置文件：

- `RSVG_DATA_ROOT`：RSVG 数据集目录
- `DIOR_DATA_ROOT`：DIOR-RSVG 数据集目录
- `OPT_DATA_ROOT`：OPT-RSVG 数据集目录
- `FLIR_DATA_ROOT`：FLIR 数据集目录
- `PRETRAINED_ROOT`：预训练权重目录

配置中的 `load_from`、`lang_model_name`、`data_root` 和 `work_dir` 可能仍有示例路径，请在运行前检查并替换为本机路径。

