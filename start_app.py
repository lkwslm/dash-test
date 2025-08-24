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
from pathlib import Path

def start_backend():
    """启动后端API服务器"""
    print("正在启动后端API服务器...")
    backend_path = Path(__file__).parent / "backend"
    os.chdir(backend_path)
    
    try:
        process = subprocess.Popen([
            sys.executable, "api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 监控后端输出
        for line in iter(process.stdout.readline, ''):
            print(f"[后端] {line.strip()}")
            if "Running on" in line:
                print("✓ 后端API服务器启动成功!")
                break
        
        return process
    except Exception as e:
        print(f"✗ 后端启动失败: {e}")
        return None

def start_frontend():
    """启动前端Dash应用"""
    print("正在启动前端Dash应用...")
    frontend_path = Path(__file__).parent / "frontend"
    os.chdir(frontend_path)
    
    try:
        process = subprocess.Popen([
            sys.executable, "dash_app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 监控前端输出
        for line in iter(process.stdout.readline, ''):
            print(f"[前端] {line.strip()}")
            if "Running on" in line:
                print("✓ 前端Dash应用启动成功!")
                break
        
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