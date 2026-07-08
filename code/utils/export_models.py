#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键导出 AI 相机所需的所有 ONNX 模型。
运行前请确保已安装：ultralytics, torch, torchvision, onnx
"""

import os
import sys

def export_yolov8n():
    """导出 YOLOv8n 目标检测模型 (320×320)"""
    print("=" * 50)
    print("正在导出 YOLOv8n 检测模型 (320×320)...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        model.export(format='onnx', imgsz=320, opset=12, simplify=True)
        print("✅ yolov8n.onnx 导出完成，文件位于当前目录。\n")
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        print("请检查 ultralytics, torch, onnx 是否已正确安装。\n")

def export_places365():
    """导出 Places365 场景分类模型 (ResNet18, 224×224)"""
    print("=" * 50)
    print("正在导出 Places365 场景分类模型 (224×224)...")
    try:
        import torch
        import torchvision

        # 方式一：尝试通过 torch.hub 自动下载（可能因网络问题失败）
        print("尝试从 torch.hub 下载预训练权重...")
        try:
            model = torch.hub.load('CSAILVision/places365', 'resnet18', pretrained=True)
        except Exception:
            # 方式二：如果自动下载失败，尝试使用本地备份
            print("自动下载失败，尝试使用本地 places365 仓库...")
            local_path = input("请输入本地 places365 仓库路径（如 C:/Users/jie/Desktop/places365-master）：").strip()
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"路径不存在: {local_path}")
            model = torch.hub.load(local_path, 'resnet18', source='local', pretrained=True)

        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        onnx_path = 'places365_resnet18.onnx'
        torch.onnx.export(model, dummy, onnx_path,
                          input_names=['input'], output_names=['output'],
                          opset_version=12)
        print(f"✅ {onnx_path} 导出完成，文件位于当前目录。\n")
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        print("请手动下载 Places365 预训练模型并转换为 ONNX，参见 models/README.md\n")

if __name__ == '__main__':
    print("AI 相机模型导出工具")
    print("=" * 50)
    export_yolov8n()
    export_places365()
    print("全部导出完毕。请将生成的 .onnx 文件复制到 models/ 目录下。")