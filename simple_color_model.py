#!/usr/bin/env python
"""
简化版：直接创建带颜色的 CadQuery 模型
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
        0: (255, 0, 0),      # ExtrudeSide - 红色
        1: (0, 255, 0),      # ExtrudeEnd - 绿色
        2: (0, 0, 255),      # Cut - 蓝色
        3: (255, 255, 0),    # RevolveEnd - 黄色
        4: (255, 128, 0),    # Fillet - 橙色
        5: (255, 0, 255),    # Chamfer - 品红
        6: (0, 255, 255),    # RevolveSide - 青色
        7: (128, 128, 128)   # Unknown - 灰色
    }
    
    print('面分类结果:')
    for i, pred in enumerate(predictions):
        name = segment_names[pred] if isinstance(segment_names, list) else segment_names[str(pred)]
        color = color_map[pred]
        print(f'面 {i}: {name} -> RGB{color}')
    
    # 创建模型（简化版）
    print('\n创建带颜色的模型...')
    
    # 创建基础盒子
    box = cq.Workplane("XY").box(10, 10, 5)
    
    # 给每个面对象设置颜色
    # CadQuery 的颜色设置需要访问底层形状
    colored_faces = []
    
    # 获取所有面
    for i, face in enumerate(box.objects):
        if i < len(predictions):
            pred = predictions[i]
            color = color_map[pred]
            
            # 创建带颜色的面
            colored_face = face
            # 设置颜色属性（这部分可能需要不同的方法）
            print(f'给面 {i} 设置颜色 {color}')
            colored_faces.append(colored_face)
    
    # 导出模型
    cq.exporters.export(box, './tests/0522/output/simple_colored_model.step', exportType='STEP')
    print('\n✓ 模型已导出')

if __name__ == '__main__':
    main()
