#!/usr/bin/env python
"""
验证 CadQuery 面与 BRepNet 面的映射关系
"""
import numpy as np
import json
import cadquery as cq

def main():
    # 加载预测结果
    logits = np.loadtxt('./tests/0522/step_data/temp_working/logits/extrude_cut_revolve.logits')
    predictions = np.argmax(logits, axis=1)
    
    # 加载类别名称
    with open('./example_files/pretrained_models/segment_names.json', 'r') as f:
        segment_names = json.load(f)
    
    # 颜色映射
    color_map = {
        0: '#FF0000',      # ExtrudeSide - 红色
        1: '#00FF00',      # ExtrudeEnd - 绿色
        2: '#0000FF',      # Cut - 蓝色
        3: '#FFFF00',      # RevolveEnd - 黄色
        4: '#FF8000',      # Fillet - 橙色
        5: '#FF00FF',      # Chamfer - 品红
        6: '#00FFFF',      # RevolveSide - 青色
        7: '#808080'       # Unknown - 灰色
    }
    
    # 创建模型
    model = cq.Workplane('XY').box(10, 10, 5).edges().fillet(1)
    
    # 获取面数量
    cq_face_count = len(model.faces().vals())
    brepnet_face_count = len(predictions)
    
    print('=' * 60)
    print('面映射验证')
    print('=' * 60)
    print(f'CadQuery 面数量: {cq_face_count}')
    print(f'BRepNet 面数量: {brepnet_face_count}')
    print(f'映射状态: {"匹配" if cq_face_count == brepnet_face_count else "不匹配"}')
    print()
    
    # 生成面映射报告
    mapping_report = []
    for i in range(min(cq_face_count, brepnet_face_count)):
        pred = predictions[i]
        name = segment_names[pred] if isinstance(segment_names, list) else segment_names[str(pred)]
        color = color_map[pred]
        
        mapping_report.append({
            'cq_face_index': i,
            'brepnet_face_index': i,
            'category': name,
            'category_index': int(pred),
            'color': color
        })
        
        print(f'面 {i}: {name} ({color})')
    
    # 保存映射报告
    with open('./tests/0522/output/face_mapping_report.json', 'w') as f:
        json.dump(mapping_report, f, indent=4)
    
    print()
    print('✓ 面映射报告已保存到: ./tests/0522/output/face_mapping_report.json')
    
    # 生成可视化 HTML
    generate_html_visualization(mapping_report, segment_names, color_map)

def generate_html_visualization(mapping_report, segment_names, color_map):
    """生成 HTML 可视化页面"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BRepNet 面分类结果</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .color-box {{ display: inline-block; width: 30px; height: 30px; margin-right: 10px; border: 1px solid #000; }}
        .face-item {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
        .legend {{ margin: 20px 0; padding: 10px; background: #e0e0e0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>BRepNet 面分类结果可视化</h1>
    <p>总面数: {len(mapping_report)}</p>
    
    <div class="legend">
        <h2>颜色图例</h2>
"""
    
    for idx, color in color_map.items():
        name = segment_names[idx] if isinstance(segment_names, list) else segment_names[str(idx)]
        html_content += f"""
        <div>
            <span class="color-box" style="background: {color};"></span>
            <span>类别 {idx}: {name}</span>
        </div>
"""
    
    html_content += """
    </div>
    
    <h2>面分类详情</h2>
"""
    
    for item in mapping_report:
        html_content += f"""
    <div class="face-item">
        <span class="color-box" style="background: {item['color']};"></span>
        <span>面 {item['cq_face_index']}: <strong>{item['category']}</strong></span>
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
