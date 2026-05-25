#!/usr/bin/env python
"""
使用 Open Cascade 给 STEP 文件中的面染色
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import json
from pathlib import Path

def color_step_file(input_step, output_step, predictions):
    """给 STEP 文件中的面染色"""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer
        from OCC.Core.STEPControl import STEPControl_AsIs
        from OCC.Extend.TopologyUtils import TopologyExplorer
        from OCC.Core.XCAFDoc import (XCAFDoc_DocumentTool_ShapeTool, 
                                     XCAFDoc_DocumentTool_ColorTool,
                                     XCAFDoc_ColorGen)
        from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCC.Core.XCAFApp import XCAFApp_Application
        from OCC.Core.TDocStd import TDocStd_Document
        
        # 颜色映射
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
        
        # 读取 STEP 文件
        reader = STEPControl_Reader()
        reader.ReadFile(str(input_step))
        reader.TransferRoots()
        shape = reader.OneShape()
        
        # 创建 XCAF 文档（修复 API）
        app = XCAFApp_Application()
        app.Init()
        doc = TDocStd_Document()
        app.NewDocument("MDTV-XCAF", doc)
        
        # 获取形状工具和颜色工具
        shape_tool = XCAFDoc_DocumentTool_ShapeTool(doc.Main())
        color_tool = XCAFDoc_DocumentTool_ColorTool(doc.Main())
        
        # 将形状添加到文档
        label = shape_tool.AddShape(shape, True)
        
        # 获取所有面
        top_exp = TopologyExplorer(shape)
        faces = list(top_exp.faces())
        
        # 给每个面设置颜色
        for i, face in enumerate(faces):
            if i < len(predictions):
                color_rgb = color_map.get(predictions[i], (0.5, 0.5, 0.5))
                color = Quantity_Color(color_rgb[0], color_rgb[1], color_rgb[2], Quantity_TOC_RGB)
                
                # 创建面的标签并设置颜色
                face_label = shape_tool.AddShape(face, False)
                color_tool.SetColor(face_label, color, XCAFDoc_ColorGen)
        
        # 使用 XCAF 格式保存
        from OCC.Core.STEPControl import STEPControl_Writer
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(output_step))
        
        print(f'✓ 带颜色的 STEP 文件已保存到: {output_step}')
        return True
        
    except Exception as e:
        print(f'✗ 设置颜色时出错: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    # 加载预测结果
    logits = np.loadtxt('./tests/0522/step_data/temp_working/logits/extrude_cut_revolve.logits')
    predictions = np.argmax(logits, axis=1)
    
    # 输入输出文件
    input_step = Path('./tests/0522/step_data/extrude_cut_revolve.step')
    output_step = Path('./tests/0522/output/colored_extrude_cut_revolve.step')
    
    # 给 STEP 文件染色
    color_step_file(input_step, output_step, predictions)

if __name__ == '__main__':
    main()
