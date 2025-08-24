from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from datetime import datetime
from defect_detector import DefectDetector
import threading
import uuid

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化缺陷检测器
detector = DefectDetector()

# 存储检测历史
detection_history = []
detection_lock = threading.Lock()

# 配置
UPLOAD_FOLDER = '../uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'defect_detection_api'
    })

@app.route('/api/detect', methods=['POST'])
def detect_defects():
    """缺陷检测接口"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'status': 'error'
            }), 400
        
        image_data = data['image']
        method = data.get('method', 'edge_detection')
        
        # 生成唯一ID
        detection_id = str(uuid.uuid4())
        
        # 执行检测
        start_time = time.time()
        result = detector.detect_defects(image_data, method)
        processing_time = time.time() - start_time
        
        # 添加额外信息
        result['detection_id'] = detection_id
        result['processing_time'] = round(processing_time, 3)
        result['timestamp'] = datetime.now().isoformat()
        
        # 保存到历史记录
        with detection_lock:
            detection_history.append({
                'id': detection_id,
                'timestamp': result['timestamp'],
                'method': method,
                'defects_found': result.get('defects_found', 0),
                'confidence_score': result.get('confidence_score', 0.0),
                'processing_time': processing_time,
                'status': result.get('status', 'unknown')
            })
            
            # 保持历史记录在合理范围内
            if len(detection_history) > 100:
                detection_history.pop(0)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/methods', methods=['GET'])
def get_detection_methods():
    """获取可用的检测方法"""
    methods = {
        'edge_detection': {
            'name': '边缘检测',
            'description': '基于Canny边缘检测算法识别缺陷',
            'suitable_for': ['裂纹', '边缘缺陷', '形状异常']
        },
        'threshold_analysis': {
            'name': '阈值分析',
            'description': '基于Otsu阈值分割识别缺陷',
            'suitable_for': ['污渍', '色差', '亮度异常']
        },
        'texture_analysis': {
            'name': '纹理分析',
            'description': '基于局部二值模式分析纹理异常',
            'suitable_for': ['表面粗糙度', '纹理不均匀']
        },
        'color_clustering': {
            'name': '颜色聚类',
            'description': '基于K-means聚类分析颜色异常',
            'suitable_for': ['色差', '颜色不均匀', '污染']
        },
        'morphological_analysis': {
            'name': '形态学分析',
            'description': '基于形态学操作检测结构缺陷',
            'suitable_for': ['孔洞', '凸起', '形状缺陷']
        },
        'comprehensive': {
            'name': '综合检测',
            'description': '结合多种方法进行全面检测',
            'suitable_for': ['全面检测', '未知缺陷类型']
        }
    }
    
    return jsonify({
        'methods': methods,
        'default_method': 'edge_detection'
    })

@app.route('/api/history', methods=['GET'])
def get_detection_history():
    """获取检测历史"""
    with detection_lock:
        # 获取最近的记录
        limit = request.args.get('limit', 20, type=int)
        history = detection_history[-limit:] if detection_history else []
        
        # 计算统计信息
        stats = detector.get_detection_statistics([
            {'defects_found': h['defects_found'], 'confidence_score': h['confidence_score']}
            for h in detection_history
        ])
        
        return jsonify({
            'history': list(reversed(history)),  # 最新的在前面
            'statistics': stats
        })

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """获取详细统计信息"""
    with detection_lock:
        if not detection_history:
            return jsonify({
                'total_detections': 0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'method_usage': {},
                'defects_trend': []
            })
        
        # 计算统计信息
        total_detections = len(detection_history)
        successful_detections = sum(1 for h in detection_history if h['status'] == 'success')
        success_rate = successful_detections / total_detections if total_detections > 0 else 0
        
        avg_processing_time = sum(h['processing_time'] for h in detection_history) / total_detections
        
        # 方法使用统计
        method_usage = {}
        for h in detection_history:
            method = h['method']
            method_usage[method] = method_usage.get(method, 0) + 1
        
        # 缺陷趋势（最近10次检测）
        recent_history = detection_history[-10:]
        defects_trend = [
            {
                'timestamp': h['timestamp'],
                'defects_found': h['defects_found'],
                'confidence': h['confidence_score']
            }
            for h in recent_history
        ]
        
        return jsonify({
            'total_detections': total_detections,
            'success_rate': round(success_rate * 100, 2),
            'avg_processing_time': round(avg_processing_time, 3),
            'method_usage': method_usage,
            'defects_trend': defects_trend
        })

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """清除检测历史"""
    with detection_lock:
        detection_history.clear()
    
    return jsonify({
        'message': 'History cleared successfully',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    print("Starting Defect Detection API Server...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/detect - Detect defects in image")
    print("  GET  /api/methods - Get available detection methods")
    print("  GET  /api/history - Get detection history")
    print("  GET  /api/statistics - Get detailed statistics")
    print("  POST /api/clear_history - Clear detection history")
    
    app.run(host='0.0.0.0', port=8080, debug=True)