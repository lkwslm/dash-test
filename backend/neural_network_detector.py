import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import os
import time
from typing import Dict, List, Any, Optional

class SimpleCNN(nn.Module):
    """最简单的CNN分类网络"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NeuralNetworkDetector:
    """神经网络缺陷检测器（最简单实现）"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_simple_model()
        self.transform = self._get_transform()
        
    def _create_simple_model(self):
        """创建简单的CNN模型"""
        model = SimpleCNN(num_classes=2)
        # 使用随机权重（最简单实现）
        return model.to(self.device)
    
    def _get_transform(self):
        """获取图像预处理转换"""
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect(self, image_data: str, model_type: str = 'cnn_classification') -> Dict[str, Any]:
        """
        使用神经网络进行缺陷检测
        
        Args:
            image_data: base64编码的图像数据
            model_type: 模型类型（目前只支持cnn_classification）
            
        Returns:
            包含检测结果的字典
        """
        start_time = time.time()
        
        try:
            # 解码图像
            image = self._decode_image(image_data)
            
            if model_type == 'cnn_classification':
                result = self._cnn_classification(image)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 添加处理时间
            result['processing_time'] = time.time() - start_time
            result['model_type'] = model_type
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'defects_found': 0,
                'confidence_score': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def _cnn_classification(self, image: np.ndarray) -> Dict[str, Any]:
        """CNN分类方法（最简单实现）"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 预处理
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 进行预测（使用随机权重，结果仅供参考）
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        confidence_score = confidence.item()
        predicted_class = predicted.item()
        
        # 模拟缺陷检测结果
        defects = []
        if predicted_class == 1:  # 假设类别1表示有缺陷
            # 生成一些模拟的缺陷位置
            height, width = image.shape[:2]
            defects.append({
                'id': 0,
                'type': 'neural_network_defect',
                'area': float(height * width * 0.1),
                'bbox': [
                    int(width * 0.2), 
                    int(height * 0.2),
                    int(width * 0.6),
                    int(height * 0.6)
                ],
                'confidence': confidence_score,
                'class_id': predicted_class,
                'class_name': 'defective' if predicted_class == 1 else 'normal'
            })
        
        # 创建可视化结果
        processed_image = self._create_visualization(image, defects)
        
        return {
            'defects_found': len(defects),
            'defects': defects,
            'confidence_score': confidence_score,
            'predictions': probabilities[0].cpu().numpy().tolist(),
            'class_id': predicted_class,
            'class_name': 'defective' if predicted_class == 1 else 'normal',
            'processed_image': self._encode_image(processed_image)
        }
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        """解码base64图像数据"""
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    def _create_visualization(self, image: np.ndarray, defects: List[Dict]) -> np.ndarray:
        """创建可视化结果"""
        vis_image = image.copy()
        
        # 绘制检测框
        for defect in defects:
            x, y, w, h = defect['bbox']
            confidence = defect['confidence']
            
            # 绘制矩形框
            color = (0, 0, 255)  # 红色
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # 添加置信度文本
            label = f"Defect: {confidence:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将图像编码为base64字符串"""
        # 转换为RGB格式
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 编码为base64
        _, buffer = cv2.imencode('.png', image_rgb)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        return [
            {
                'name': 'simple_cnn',
                'type': 'cnn_classification',
                'description': '简单的CNN分类模型',
                'input_size': '64x64',
                'classes': ['normal', 'defective']
            }
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        if model_name == 'simple_cnn':
            return {
                'name': 'simple_cnn',
                'type': 'cnn_classification',
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device),
                'input_shape': [1, 3, 64, 64],
                'output_classes': 2
            }
        else:
            raise ValueError(f"未知模型: {model_name}")

# 全局实例
neural_detector = NeuralNetworkDetector()
