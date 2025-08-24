#!/usr/bin/env python3
"""
API测试脚本
测试后端API的各个端点
"""

import requests
import json
import base64
import numpy as np
from PIL import Image
import io

API_BASE_URL = "http://localhost:8080/api"

def create_test_image():
    """创建一个测试图像"""
    # 创建一个简单的测试图像（带有一些"缺陷"）
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 添加一些"缺陷"
    img[50:70, 50:70] = [0, 0, 0]  # 黑色方块
    img[120:140, 120:140] = [255, 0, 0]  # 红色方块
    img[80:100, 150:170] = [0, 255, 0]  # 绿色方块
    
    # 转换为PIL图像
    pil_img = Image.fromarray(img)
    
    # 转换为base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def test_health():
    """测试健康检查端点"""
    print("测试健康检查...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_methods():
    """测试获取检测方法端点"""
    print("\n测试获取检测方法...")
    try:
        response = requests.get(f"{API_BASE_URL}/methods")
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"可用方法数量: {len(data['methods'])}")
        for method, info in data['methods'].items():
            print(f"  - {method}: {info['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"获取方法失败: {e}")
        return False

def test_detection():
    """测试缺陷检测端点"""
    print("\n测试缺陷检测...")
    
    # 创建测试图像
    test_image = create_test_image()
    
    # 测试不同的检测方法
    methods = ['edge_detection', 'threshold_analysis', 'texture_analysis']
    
    for method in methods:
        print(f"\n测试方法: {method}")
        try:
            response = requests.post(f"{API_BASE_URL}/detect", json={
                'image': test_image,
                'method': method
            })
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"检测状态: {result['status']}")
                print(f"发现缺陷: {result.get('defects_found', 0)} 个")
                print(f"置信度: {result.get('confidence_score', 0):.2%}")
                print(f"处理时间: {result.get('processing_time', 0):.3f} 秒")
            else:
                print(f"检测失败: {response.text}")
                
        except Exception as e:
            print(f"检测请求失败: {e}")

def test_history():
    """测试历史记录端点"""
    print("\n测试历史记录...")
    try:
        response = requests.get(f"{API_BASE_URL}/history")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"历史记录数量: {len(data['history'])}")
            print(f"统计信息: {data['statistics']}")
        else:
            print(f"获取历史失败: {response.text}")
            
    except Exception as e:
        print(f"历史记录请求失败: {e}")

def test_statistics():
    """测试统计信息端点"""
    print("\n测试统计信息...")
    try:
        response = requests.get(f"{API_BASE_URL}/statistics")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"总检测次数: {stats['total_detections']}")
            print(f"成功率: {stats['success_rate']:.1f}%")
            print(f"平均处理时间: {stats['avg_processing_time']:.3f}s")
            print(f"方法使用统计: {stats['method_usage']}")
        else:
            print(f"获取统计失败: {response.text}")
            
    except Exception as e:
        print(f"统计信息请求失败: {e}")

def main():
    print("=" * 50)
    print("缺陷检测API测试")
    print("=" * 50)
    
    # 测试各个端点
    tests = [
        ("健康检查", test_health),
        ("检测方法", test_methods),
        ("缺陷检测", test_detection),
        ("历史记录", test_history),
        ("统计信息", test_statistics)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}测试出现异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 测试通过")

if __name__ == "__main__":
    main()