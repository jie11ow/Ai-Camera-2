cat > models/README.md << 'EOF'
# AI 模型来源与版权说明

本目录包含两个预训练 ONNX 模型，用于边缘端推理。模型文件本身**不在 Git 仓库中**，请通过下载脚本获取。

## 1. 目标检测模型：YOLOv8n
- **来源**：[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **预训练数据**：COCO 2017 train (80 类)
- **原始权重**：`yolov8n.pt`，由 Ultralytics 使用官方代码训练
- **转换为 ONNX**：使用 Ultralytics 库导出（320×320, opset 12, simplify=True）
- **协议**：**AGPL-3.0**（模型权重继承自 Ultralytics，请参考 [Ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)）
- **能力范围**：可检测日常场景中常见的 80 类物体

## 2. 场景分类模型：Places365 (ResNet18)
- **来源**：[MIT CSAIL Places365](http://places2.csail.mit.edu/)
- **预训练数据**：Places365-Standard (365 类场景)
- **模型架构**：ResNet18，在 Places365 上训练，原始论文为 *Places: A 10 million Image Database for Scene Recognition*
- **转换为 ONNX**：使用 PyTorch 导出（224×224, opset 12）
- **协议**：**MIT**（Places365 数据集及预训练模型均以 MIT 协议发布，允许自由使用）
- **能力范围**：可识别 365 种场景，本项目中通过 `scene_mapping_places365.json` 映射为 10+ 个摄影场景（如室内、海滩、城市、夜景等）

## 如何获取模型？
请运行同目录下的 `download_models.py`（Python 版，跨平台）或 `download_models.sh`（Linux），脚本会自动从 GitHub Releases 下载相应的 `.onnx` 文件。

如果下载失败，可前往 [GitHub Releases 页面](https://github.com/你的用户名/AI-Camera-K1/releases) 手动下载。
EOF