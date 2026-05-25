#!/usr/bin/env python
"""
处理STEP文件的脚本
使用方法：
    python process_step.py
"""

import sys
from pathlib import Path
import json
import numpy as np

# 添加BRepNet到路径
brepnet_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brepnet_root))

from pipeline.extract_brepnet_data_from_step import BRepNetExtractor
import utils.data_utils as data_utils


def main():
    # 设置路径
    step_file = Path(__file__).parent / "step_data" / "extrude_cut_revolve.step"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 加载特征schema
    feature_schema_path = brepnet_root / "feature_lists" / "all.json"
    with open(feature_schema_path, "r") as fp:
        feature_schema = json.load(fp)

    print(f"处理STEP文件: {step_file.name}")
    print(f"输出目录: {output_dir}")

    # 创建提取器并处理
    extractor = BRepNetExtractor(
        step_file,
        output_dir,
        feature_schema,
        scale_body=True  # 将模型缩放到单位盒[-1,1]^3
    )

    print("开始提取特征...")
    extractor.process()

    # 加载并分析结果
    npz_file = output_dir / f"{step_file.stem}.npz"
    if npz_file.exists():
        print(f"\n成功生成: {npz_file.name}")
        data = data_utils.load_npz_data(npz_file)

        print("\n" + "="*60)
        print("提取结果摘要")
        print("="*60)

        # 基本统计
        num_faces = data['face_features'].shape[0]
        num_edges = data['edge_features'].shape[0]
        num_coedges = data['coedge_features'].shape[0]

        print(f"面数量: {num_faces}")
        print(f"边数量: {num_edges}")
        print(f"半边数量: {num_coedges}")

        # 面类型统计
        face_features = data['face_features']
        face_types = ['Plane', 'Cylinder', 'Cone', 'Sphere', 'Torus']
        print("\n面类型统计:")
        for i, face_type in enumerate(face_types):
            count = np.sum(face_features[:, i] > 0.5)
            if count > 0:
                print(f"  {face_type}: {count}")

        # 边类型统计
        edge_features = data['edge_features']
        print("\n边的凹凸性:")
        concave = np.sum(edge_features[:, 0] > 0.5)
        convex = np.sum(edge_features[:, 1] > 0.5)
        smooth = np.sum(edge_features[:, 2] > 0.5)
        print(f"  凹边: {concave}")
        print(f"  凸边: {convex}")
        print(f"  光滑边: {smooth}")

        # 边曲线类型
        edge_curve_types = ['Circular', 'Closed', 'Elliptical', 'NonRationalBSpline',
                           'RationalBSpline', 'Straight']
        print("\n边的曲线类型:")
        for i, curve_type in enumerate(edge_curve_types):
            count = np.sum(edge_features[:, i+4] > 0.5)
            if count > 0:
                print(f"  {curve_type}: {count}")

        # 文件大小
        file_size_kb = npz_file.stat().st_size / 1024
        print(f"\n输出文件大小: {file_size_kb:.2f} KB")
        print("="*60)

    else:
        print(f"\n错误: 未能生成输出文件")


if __name__ == '__main__':
    main()
