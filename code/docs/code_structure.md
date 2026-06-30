# 代码结构说明

## 核心程序
- **main.py**：主程序入口，负责摄像头采集、UI 显示、按键处理、调用 AI 推理线程。
- **ai_engine.py**（若有分离）：包含 YOLOv8n 目标检测、Places365 场景分类、构图规则分析的函数。
- **gpio.py**（若有分离）：GPIO 初始化、按钮轮询相关函数。

## 辅助资源
- **utils/corpus_interactive.json**：智能构图建议语料库，由 `composition_hints()` 读取。
- **utils/scene_mapping_places365.json**：Places365 365 类到摄影场景的映射表。

## 工具脚本
- **tools/setup_gpio.sh**：初始化 GPIO 引脚权限的 Shell 脚本。
- **tools/start_ai_camera.sh**：自启动入口脚本，等待图形界面就绪后启动主程序。
- **tools/run_ai_camera.service**：systemd 用户服务文件模板。

## 模型导出
- **utils/export_models.py**：在 PC 上运行，自动下载预训练权重并导出 ONNX 模型。
- **models/download_models.sh**：从网络下载 ONNX 模型到本地 models/ 目录。

## 文档
- **docs/user_manual.md**：用户操作手册。
- **docs/technical_report.md**：详细技术报告。