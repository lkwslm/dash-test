import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import filters, measure, morphology
from skimage.feature import local_binary_pattern
import pandas as pd
from typing import Dict, List, Tuple, Any
import base64
import io
from PIL import Image

class DefectDetector:
    """缺陷检测算法类"""
    
    def __init__(self):
        self.detection_methods = {
            'edge_detection': self._edge_detection,
            'threshold_analysis': self._threshold_analysis,
            'texture_analysis': self._texture_analysis,
            'color_clustering': self._color_clustering,
            'morphological_analysis': self._morphological_analysis
        }
    
    def detect_defects(self, image_data: str, method: str = 'edge_detection') -> Dict[str, Any]:
        """
        主要的缺陷检测函数
        
        Args:
            image_data: base64编码的图像数据
            method: 检测方法
            
        Returns:
            包含检测结果的字典
        """
        try:
            # 解码图像
            image = self._decode_image(image_data)
            
            # 执行检测
            if method in self.detection_methods:
                result = self.detection_methods[method](image)
            else:
                result = self._comprehensive_detection(image)
            
            # 添加基本信息
            result['image_shape'] = image.shape
            result['method_used'] = method
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'defects_found': 0,
                'confidence_score': 0.0
            }
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        """解码base64图像数据"""
        # 移除data:image前缀
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # 解码base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 如果是RGB，转换为BGR（OpenCV格式）
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    def _edge_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """边缘检测方法"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析轮廓
        defects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # 过滤小的噪声
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'id': i,
                    'type': 'edge_anomaly',
                    'area': float(area),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': min(0.9, area / 1000)
                })
        
        # 计算整体置信度
        confidence = min(0.95, len(defects) * 0.1 + 0.5) if defects else 0.1
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'confidence_score': confidence,
            'processed_image': self._encode_image(edges)
        }
    
    def _threshold_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """阈值分析方法"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Otsu阈值
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
        
        defects = []
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50:
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                defects.append({
                    'id': i-1,
                    'type': 'threshold_anomaly',
                    'area': float(area),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': min(0.9, area / 500)
                })
        
        confidence = min(0.9, len(defects) * 0.15 + 0.4) if defects else 0.2
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'confidence_score': confidence,
            'processed_image': self._encode_image(cleaned)
        }
    
    def _texture_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """纹理分析方法"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 局部二值模式
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # 计算LBP直方图
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # 检测纹理异常
        # 这里使用简单的统计方法
        texture_variance = np.var(lbp)
        texture_mean = np.mean(lbp)
        
        # 基于纹理特征判断缺陷
        defects = []
        if texture_variance > np.percentile(lbp, 95):
            defects.append({
                'id': 0,
                'type': 'texture_anomaly',
                'area': float(gray.shape[0] * gray.shape[1] * 0.1),
                'bbox': [0, 0, gray.shape[1], gray.shape[0]],
                'confidence': min(0.8, texture_variance / 1000)
            })
        
        confidence = 0.7 if defects else 0.3
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'confidence_score': confidence,
            'processed_image': self._encode_image((lbp * 255 / lbp.max()).astype(np.uint8))
        }
    
    def _color_clustering(self, image: np.ndarray) -> Dict[str, Any]:
        """颜色聚类方法"""
        if len(image.shape) != 3:
            # 如果是灰度图，转换为3通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 重塑图像数据
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means聚类
        k = 4
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 重塑标签
        labels = labels.reshape(image.shape[:2])
        
        # 分析每个聚类
        defects = []
        for i in range(k):
            cluster_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 200:  # 只考虑较大的区域
                    x, y, w, h = cv2.boundingRect(contour)
                    defects.append({
                        'id': i * 100 + j,
                        'type': f'color_cluster_{i}',
                        'area': float(area),
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': min(0.8, area / 1000)
                    })
        
        confidence = min(0.85, len(defects) * 0.1 + 0.3) if defects else 0.2
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'confidence_score': confidence,
            'processed_image': self._encode_image((labels * 255 / k).astype(np.uint8))
        }
    
    def _morphological_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """形态学分析方法"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 开运算（去除噪声）
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # 闭运算（填充孔洞）
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # 梯度运算（边缘检测）
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 阈值处理
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({
                    'id': i,
                    'type': 'morphological_anomaly',
                    'area': float(area),
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': min(0.9, area / 800)
                })
        
        confidence = min(0.9, len(defects) * 0.12 + 0.4) if defects else 0.25
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'confidence_score': confidence,
            'processed_image': self._encode_image(thresh)
        }
    
    def _comprehensive_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """综合检测方法"""
        # 运行多种检测方法
        methods = ['edge_detection', 'threshold_analysis', 'texture_analysis']
        all_results = {}
        all_defects = []
        
        for method in methods:
            result = self.detection_methods[method](image)
            all_results[method] = result
            all_defects.extend(result['defects'])
        
        # 合并结果
        total_defects = len(all_defects)
        avg_confidence = np.mean([all_results[m]['confidence_score'] for m in methods])
        
        return {
            'defects_found': total_defects,
            'defects': all_defects,
            'confidence_score': float(avg_confidence),
            'method_results': all_results,
            'processed_image': all_results['edge_detection']['processed_image']
        }
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将图像编码为base64字符串"""
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def get_detection_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取检测统计信息"""
        if not results:
            return {'total_images': 0, 'total_defects': 0, 'avg_confidence': 0.0}
        
        total_defects = sum(r.get('defects_found', 0) for r in results)
        avg_confidence = np.mean([r.get('confidence_score', 0) for r in results])
        
        defect_types = {}
        for result in results:
            for defect in result.get('defects', []):
                defect_type = defect.get('type', 'unknown')
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
        
        return {
            'total_images': len(results),
            'total_defects': total_defects,
            'avg_confidence': float(avg_confidence),
            'defect_types': defect_types,
            'avg_defects_per_image': total_defects / len(results) if results else 0
        }