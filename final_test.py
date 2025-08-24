#!/usr/bin/env python3
"""
智能缺陷检测系统 - 最终测试脚本
测试前后端通信和所有功能
"""

import requests
import base64
import json
import time
from PIL import Image
import numpy as np
import io

# 配置
API_BASE_URL = "http://localhost:8080/api"
FRONTEND_URL = "http://localhost:12000"

def test_backend_health():
    """测试后端健康状态"""
    print("🔍 测试后端健康状态...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 后端健康: {data['status']}")
            return True
        else:
            print(f"❌ 后端健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 后端连接失败: {e}")
        return False

def test_frontend_health():
    """测试前端健康状态"""
    print("🔍 测试前端健康状态...")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print("✅ 前端正常运行")
            return True
        else:
            print(f"❌ 前端访问失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 前端连接失败: {e}")
        return False

def create_test_image():
    """创建测试图像"""
    img = np.ones((150, 150, 3), dtype=np.uint8) * 255
    
    # 添加缺陷
    img[20:40, 20:40] = [0, 0, 0]  # 黑色方块
    img[100:120, 100:120] = [255, 0, 0]  # 红色方块
    img[60:65, 30:120] = [0, 255, 0]  # 绿色线条
    
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return buffer.getvalue()

def test_detection_methods():
    """测试所有检测方法"""
    print("🔍 测试缺陷检测方法...")
    
    # 获取可用方法
    try:
        response = requests.get(f"{API_BASE_URL}/methods", timeout=5)
        if response.status_code != 200:
            print("❌ 无法获取检测方法")
            return False
        
        methods_data = response.json()
        methods = methods_data['methods']
        print(f"✅ 获取到 {len(methods)} 种检测方法")
        
        # 创建测试图像
        image_data = create_test_image()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        # 测试每种方法
        success_count = 0
        for method_id, method_info in methods.items():
            try:
                response = requests.post(
                    f"{API_BASE_URL}/detect",
                    json={
                        'image': image_data_url,
                        'method': method_id
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['status'] == 'success':
                        print(f"  ✅ {method_info['name']}: {result['defects_found']} 个缺陷")
                        success_count += 1
                    else:
                        print(f"  ❌ {method_info['name']}: 检测失败")
                else:
                    print(f"  ❌ {method_info['name']}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ {method_info['name']}: {e}")
        
        print(f"✅ 成功测试 {success_count}/{len(methods)} 种方法")
        return success_count == len(methods)
        
    except Exception as e:
        print(f"❌ 检测方法测试失败: {e}")
        return False

def test_statistics():
    """测试统计功能"""
    print("🔍 测试统计功能...")
    try:
        response = requests.get(f"{API_BASE_URL}/statistics", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ 统计信息: {stats['total_detections']} 次检测, 成功率 {stats['success_rate']:.1f}%")
            return True
        else:
            print(f"❌ 统计信息获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 统计功能测试失败: {e}")
        return False

def test_history():
    """测试历史记录功能"""
    print("🔍 测试历史记录功能...")
    try:
        response = requests.get(f"{API_BASE_URL}/history", timeout=5)
        if response.status_code == 200:
            history = response.json()
            print(f"✅ 历史记录: {len(history)} 条记录")
            return True
        else:
            print(f"❌ 历史记录获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 历史记录测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 智能缺陷检测系统 - 最终测试")
    print("=" * 60)
    
    tests = [
        ("后端健康检查", test_backend_health),
        ("前端健康检查", test_frontend_health),
        ("缺陷检测方法", test_detection_methods),
        ("统计功能", test_statistics),
        ("历史记录功能", test_history),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        time.sleep(1)  # 避免请求过快
    
    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    print(f"通过: {passed}/{total} 项测试")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常")
        print("\n🌐 访问地址:")
        print(f"   前端界面: {FRONTEND_URL}")
        print(f"   后端API: {API_BASE_URL}")
        print("\n📝 使用说明:")
        print("   1. 打开前端界面上传图像")
        print("   2. 选择检测方法")
        print("   3. 查看检测结果和统计信息")
        print("   4. 使用API进行程序化调用")
    else:
        print(f"⚠️  {total - passed} 项测试失败，请检查系统状态")
    
    print("=" * 60)

if __name__ == "__main__":
    main()