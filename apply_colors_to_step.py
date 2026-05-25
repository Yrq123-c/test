#!/usr/bin/env python
"""
给 STEP 文件中的面根据 BRepNet 预测结果染色
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import json
from pathlib import Path

def main():
    # 1. 加载预测结果
    logits = np.loadtxt('./tests/0522/step_data/temp_working/logits/extrude_cut_revolve.logits')
    predictions = np.argmax(logits, axis=1)
    
    # 2. 加载类别名称
    with open('./example_files/pretrained_models/segment_names.json', 'r') as f:
        segment_names = json.load(f)
    
    # 3. 定义颜色映射（RGB 0-1）
    color_map = {
        0: (1.0, 0.0, 0.0),      # ExtrudeSide - 红色
        1: (0.0, 1.0, 0.0),      # ExtrudeEnd - 绿色
        2: (0.0, 0.0, 1.0),      # Cut - 蓝色
        3: (1.0, 1.0, 0.0),      # RevolveEnd - 黄色
        4: (1.0, 0.5, 0.0),      # Fillet - 橙色
        5: (1.0, 0.0, 1.0),      # Chamfer - 品红
        6: (0.0, 1.0, 1.0),      # RevolveSide - 青色
        7: (0.5, 0.5, 0.5)       # Unknown - 灰色
    }
    
    # 4. 打印分类结果和颜色
    print('=' * 60)
    print('面分类结果和颜色映射')
    print('=' * 60)
    
    for i, pred in enumerate(predictions):
        name = segment_names[pred] if isinstance(segment_names, list) else segment_names[str(pred)]
        color = color_map[pred]
        print(f'面 {i:2d}: {name:15s} -> RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})')
    
    # 5. 保存颜色映射文件
    color_mapping = {
        'face_count': len(predictions),
        'predictions': predictions.tolist(),
        'category_names': segment_names if isinstance(segment_names, list) else list(segment_names.values()),
        'color_map': {str(k): list(v) for k, v in color_map.items()}
    }
    
    with open('./tests/0522/output/face_color_mapping.json', 'w') as f:
        json.dump(color_mapping, f, indent=4)
    
    print('\\n✓ 颜色映射文件已保存到: ./tests/0522/output/face_color_mapping.json')
    
    # 6. 生成可视化 HTML
    generate_visualization_html(predictions, segment_names, color_map)

def generate_visualization_html(predictions, segment_names, color_map):
    """生成一个简单的 HTML 可视化页面"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BRepNet 面分类结果</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .color-box {{ display: inline-block; width: 30px; height: 30px; margin-right: 10px; border: 1px solid #000; }}
        .face-item {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>BRepNet 面分类结果可视化</h1>
    <p>总面数: {len(predictions)}</p>
    
    <h2>颜色图例</h2>
"""
    
    for idx, (cat_idx, color) in enumerate(color_map.items()):
        name = segment_names[cat_idx] if isinstance(segment_names, list) else segment_names[str(cat_idx)]
        html_content += f"""
    <div>
        <span class="color-box" style="background: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)});"></span>
        <span>类别 {cat_idx}: {name}</span>
    </div>
"""
    
    html_content += """
    <h2>面分类详情</h2>
"""
    
    for i, pred in enumerate(predictions):
        name = segment_names[pred] if isinstance(segment_names, list) else segment_names[str(pred)]
        color = color_map[pred]
        html_content += f"""
    <div class="face-item">
        <span class="color-box" style="background: rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)});"></span>
        <span>面 {i}: <strong>{name}</strong></span>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open('./tests/0522/output/face_classification.html', 'w') as f:
        f.write(html_content)
    
    print('✓ 可视化 HTML 已保存到: ./tests/0522/output/face_classification.html')

if __name__ == '__main__':
    main()
