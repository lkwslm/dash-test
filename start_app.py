#!/usr/bin/env python3
"""
缺陷检测应用启动脚本
同时启动后端API服务器和前端Dash应用
"""

import subprocess
import time
import sys
import os
import signal
import threading
import socket
from pathlib import Path

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def start_backend():
    """启动后端API服务器"""
    print("正在启动后端API服务器...")
    backend_path = Path(__file__).parent / "backend"
    
    # 检查端口是否被占用
    backend_port = 8080
    if is_port_in_use(backend_port):
        print(f"⚠ 端口 {backend_port} 已被占用，尝试使用备用端口...")
        backend_port = 8081
        if is_port_in_use(backend_port):
            print(f"✗ 备用端口 {backend_port} 也被占用，无法启动后端")
            return None
    
    try:
        process = subprocess.Popen([
            sys.executable, "api_server.py", "--port", str(backend_port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(backend_path))
        
        # 非阻塞方式读取输出
        def read_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(f"[后端] {line.strip()}")
                        if "Running on" in line:
                            print("✓ 后端API服务器启动成功!")
                            break
            except Exception as e:
                print(f"[后端输出读取错误] {e}")
        
        # 启动输出读取线程
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()
        
        # 等待一段时间让服务器启动，并检查进程状态
        time.sleep(3)
        if process.poll() is not None:
            # 进程已退出，读取错误输出
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"[后端错误] {stderr_output}")
            return None
        
        # 检查API是否真的可用
        try:
            import requests
            response = requests.get(f"http://localhost:{backend_port}/api/health", timeout=5)
            if response.status_code == 200:
                print("✓ 后端API服务器启动成功且健康检查通过!")
            else:
                print(f"⚠ 后端API健康检查失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠ 后端API健康检查异常: {e}")
        
        return process
    except Exception as e:
        print(f"✗ 后端启动失败: {e}")
        return None

def start_frontend():
    """启动前端Dash应用"""
    print("正在启动前端Dash应用...")
    frontend_path = Path(__file__).parent / "frontend"
    
    # 检查端口是否被占用
    frontend_port = 12000
    if is_port_in_use(frontend_port):
        print(f"⚠ 端口 {frontend_port} 已被占用，尝试使用备用端口...")
        frontend_port = 12001
        if is_port_in_use(frontend_port):
            print(f"✗ 备用端口 {frontend_port} 也被占用，无法启动前端")
            return None
    
    try:
        process = subprocess.Popen([
            sys.executable, "dash_app.py", "--port", str(frontend_port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(frontend_path))
        
        # 非阻塞方式读取输出
        def read_output():
            for line in iter(process.stdout.readline, ''):
                print(f"[前端] {line.strip()}")
                if "Running on" in line:
                    print("✓ 前端Dash应用启动成功!")
                    break
        
        # 启动输出读取线程
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()
        
        # 等待一段时间让应用启动
        time.sleep(2)
        
        return process
    except Exception as e:
        print(f"✗ 前端启动失败: {e}")
        return None

def monitor_process(process, name):
    """监控进程状态"""
    while True:
        if process.poll() is not None:
            print(f"⚠ {name}进程已退出")
            break
        time.sleep(1)

def main():
    print("=" * 60)
    print("智能缺陷检测系统启动器")
    print("=" * 60)
    
    processes = []
    
    try:
        # 启动后端
        backend_process = start_backend()
        if backend_process:
            processes.append(('后端API', backend_process))
            time.sleep(3)  # 等待后端完全启动
        else:
            print("后端启动失败，退出...")
            return
        
        # 启动前端
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(('前端Dash', frontend_process))
            time.sleep(2)  # 等待前端启动
        else:
            print("前端启动失败，但后端仍在运行...")
        
        print("\n" + "=" * 60)
        print("系统启动完成!")
        print("=" * 60)
        print("访问地址:")
        print("  前端界面: https://work-1-qnumwlwubkuzlrtf.prod-runtime.all-hands.dev")
        print("  后端API:  https://work-2-qnumwlwubkuzlrtf.prod-runtime.all-hands.dev")
        print("\n可用的API端点:")
        print("  GET  /api/health - 健康检查")
        print("  POST /api/detect - 缺陷检测")
        print("  GET  /api/methods - 获取检测方法")
        print("  GET  /api/history - 获取检测历史")
        print("  GET  /api/statistics - 获取统计信息")
        print("  POST /api/clear_history - 清除历史记录")
        print("\n按 Ctrl+C 停止所有服务")
        print("=" * 60)
        
        # 启动监控线程
        for name, process in processes:
            monitor_thread = threading.Thread(target=monitor_process, args=(process, name))
            monitor_thread.daemon = True
            monitor_thread.start()
        
        # 等待用户中断
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在停止所有服务...")
        
        # 停止所有进程
        for name, process in processes:
            try:
                print(f"停止{name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"✓ {name}已停止")
            except subprocess.TimeoutExpired:
                print(f"强制停止{name}...")
                process.kill()
            except Exception as e:
                print(f"停止{name}时出错: {e}")
        
        print("所有服务已停止")

if __name__ == "__main__":
    main()
