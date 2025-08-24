#!/usr/bin/env python3
"""
神经网络API测试脚本
用于测试新添加的神经网络检测接口
"""

import requests
import json
import base64
import os
from PIL import Image
import numpy as np

# API基础URL
API_BASE = "http://localhost:8080/api"

def test_health():
    """测试健康检查接口"""
    print("测试健康检查接口...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_get_neural_models():
    """测试获取神经网络模型列表"""
    print("\n测试获取神经网络模型列表...")
    try:
        response = requests.get(f"{API_BASE}/neural/models")
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"可用模型: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return False

def test_get_model_info():
    """测试获取模型详细信息"""
    print("\n测试获取模型详细信息...")
    try:
        response = requests.get(f"{API_BASE}/neural/models/simple_cnn")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"模型信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        return False

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # 保存到内存
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_str}"

def test_neural_detection():
    """测试神经网络检测接口"""
    print("\n测试神经网络检测接口...")
    try:
        # 创建测试图像
        test_image = create_test_image()
        
        # 准备请求数据
        payload = {
            "image": test_image,
            "model_type": "cnn_classification",
            "model_name": "simple_cnn"
        }
        
        response = requests.post(f"{API_BASE}/neural/detect", json=payload)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("检测结果摘要:")
            print(f"  状态: {data.get('status', 'unknown')}")
            print(f"  发现缺陷: {data.get('defects_found', 0)} 个")
            print(f"  置信度: {data.get('confidence_score', 0.0):.3f}")
            print(f"  处理时间: {data.get('processing_time', 0.0):.3f} 秒")
            print(f"  模型类型: {data.get('model_type', 'unknown')}")
            
            # 显示缺陷详情（如果有）
            defects = data.get('defects', [])
            if defects:
                print("  缺陷详情:")
                for defect in defects:
                    print(f"    - ID: {defect.get('id')}, 类型: {defect.get('type')}, "
                          f"置信度: {defect.get('confidence', 0.0):.3f}")
            
            return True
        else:
            print(f"错误响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"神经网络检测失败: {e}")
        return False

def test_neural_statistics():
    """测试神经网络统计接口"""
    print("\n测试神经网络统计接口...")
    try:
        response = requests.get(f"{API_BASE}/neural/statistics")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"统计信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"获取统计信息失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试神经网络API接口...")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_health,
        test_get_neural_models,
        test_get_model_info,
        test_neural_detection,
        test_neural_statistics
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"测试结果: {'通过' if result else '失败'}")
            print("-" * 30)
        except Exception as e:
            print(f"测试异常: {e}")
            results.append(False)
            print("-" * 30)
    
    # 汇总结果
    passed = sum(results)
    total = len(results)
    print(f"\n测试完成: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("所有测试通过！神经网络API接口正常工作。")
        return True
    else:
        print("部分测试失败，请检查API服务器是否正常运行。")
        return False

if __name__ == "__main__":
    # 检查API服务器是否运行
    try:
        requests.get(f"{API_BASE}/health", timeout=2)
        print("检测到API服务器正在运行，开始测试...")
    except:
        print("警告: API服务器可能未运行，请先启动后端服务器:")
        print("cd backend && python api_server.py")
        exit(1)
    
    success = main()
    exit(0 if success else 1)
