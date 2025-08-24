# 智能缺陷检测系统

基于Dash和Flask构建的智能缺陷检测应用，支持传统图像处理算法和神经网络算法，使用HTTP通信实现前后端分离架构。

## 系统架构

```
┌─────────────────┐    HTTP/REST API   ┌─────────────────┐
│   Dash 前端     │ ◄─────────────────► │   Flask 后端    │
│   (端口 12000)  │                    │   (端口 8080)   │
└─────────────────┘                    └─────────────────┘
        │                                       │
        ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│   用户界面      │                    │   检测算法引擎   │
│ - 图像上传      │                    │ - 6种传统算法   │
│ - 结果展示      │                    │ - 神经网络算法  │
│ - 历史记录      │                    │ - 统计分析      │
│ - 统计图表      │                    │ - 历史管理      │
│ - 神经网络选项  │                    │ - 性能监控      │
└─────────────────┘                    └─────────────────┘
```

## 功能特性

### 前端功能 (Dash)
- 🖼️ **图像上传**: 支持拖拽上传图像文件
- 🔍 **检测方法选择**: 6种不同的缺陷检测算法
- 📊 **实时结果展示**: 原始图像和处理结果对比
- 📈 **统计图表**: 检测趋势和统计信息可视化
- 📝 **历史记录**: 检测历史和详细信息
- 🔄 **实时监控**: 系统状态和性能监控

### 后端功能 (Flask API)
- 🧠 **多种检测算法**:
  - 边缘检测 (Canny算法)
  - 阈值分析 (Otsu阈值)
  - 纹理分析 (局部二值模式)
  - 颜色聚类 (K-means)
  - 形态学分析
  - 综合检测
  - 神经网络检测 (CNN分类)
- 📊 **统计分析**: 检测结果统计和性能分析
- 💾 **历史管理**: 检测历史存储和管理
- 🔌 **RESTful API**: 标准化的API接口
- 🤖 **神经网络支持**: PyTorch深度学习框架
- 📈 **模型管理**: 神经网络模型管理和扩展

## 安装和运行

### 环境要求
- Python 3.8+
- 所需依赖包（见 requirements.txt）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动应用

#### 方法1: 使用启动脚本（推荐）
```bash
python start_app.py
```

#### 方法2: 分别启动前后端
```bash
# 启动后端API服务器
cd backend
python api_server.py

# 启动前端Dash应用
cd frontend
python dash_app.py
```

### 访问地址
- **前端界面**: http://localhost:12000
- **后端API**: http://localhost:8080

## API接口文档

### 健康检查
```
GET /api/health
```

### 缺陷检测
```
POST /api/detect
Content-Type: application/json

{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "method": "edge_detection"
}
```

### 神经网络检测
```
POST /api/neural/detect
Content-Type: application/json

{
    "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "model_type": "cnn_classification",
    "model_name": "simple_cnn"
}
```

### 获取检测方法
```
GET /api/methods
```

### 获取神经网络模型
```
GET /api/neural/models
```

### 获取模型详情
```
GET /api/neural/models/{model_name}
```

### 获取检测历史
```
GET /api/history?limit=10
```

### 获取统计信息
```
GET /api/statistics
```

### 获取神经网络统计
```
GET /api/neural/statistics
```

### 清除历史记录
```
POST /api/clear_history
```

## 检测算法说明

### 1. 边缘检测 (edge_detection)
- **原理**: 使用Canny边缘检测算法
- **适用**: 裂纹、边缘缺陷、形状异常
- **特点**: 高精度边缘识别

### 2. 阈值分析 (threshold_analysis)
- **原理**: 基于Otsu阈值分割
- **适用**: 污渍、色差、亮度异常
- **特点**: 快速二值化处理

### 3. 纹理分析 (texture_analysis)
- **原理**: 局部二值模式(LBP)
- **适用**: 表面粗糙度、纹理不均匀
- **特点**: 纹理特征提取

### 4. 颜色聚类 (color_clustering)
- **原理**: K-means聚类算法
- **适用**: 色差、颜色不均匀、污染
- **特点**: 颜色区域分割

### 5. 形态学分析 (morphological_analysis)
- **原理**: 形态学操作(开运算、闭运算、梯度)
- **适用**: 孔洞、凸起、形状缺陷
- **特点**: 结构特征分析

### 6. 综合检测 (comprehensive)
- **原理**: 结合多种检测方法
- **适用**: 全面检测、未知缺陷类型
- **特点**: 高覆盖率检测

## 项目结构

```
project/
├── backend/                 # 后端代码
│   ├── api_server.py       # Flask API服务器
│   ├── defect_detector.py  # 传统缺陷检测算法
│   └── neural_network_detector.py  # 神经网络检测算法
├── frontend/               # 前端代码
│   └── dash_app.py        # Dash应用界面
├── assets/                 # 静态资源
├── uploads/               # 上传文件目录
├── requirements.txt       # 依赖包列表
├── start_app.py          # 应用启动脚本
├── test_api.py           # API测试脚本
├── test_neural_api.py    # 神经网络API测试脚本
└── README.md             # 项目说明
```

## 测试

运行API测试脚本：
```bash
python test_api.py
```

运行神经网络API测试脚本：
```bash
python test_neural_api.py
```

## 技术栈

- **前端**: Dash, Plotly, Bootstrap
- **后端**: Flask, OpenCV, scikit-image, scikit-learn
- **图像处理**: OpenCV, PIL, NumPy
- **机器学习**: scikit-learn, scikit-image
- **深度学习**: PyTorch, torchvision
- **通信**: HTTP/REST API, JSON

## 特性亮点

1. **模块化设计**: 前后端完全分离，便于维护和扩展
2. **多算法支持**: 6种不同的缺陷检测算法，适应不同场景
3. **实时交互**: 基于Dash的响应式用户界面
4. **性能监控**: 实时系统状态和性能统计
5. **历史管理**: 完整的检测历史记录和统计分析
6. **易于扩展**: 标准化的API接口，便于集成新算法

## 开发说明

### 添加新的检测算法
1. 在 `DefectDetector` 类中添加新方法
2. 在 `detection_methods` 字典中注册
3. 更新API文档和前端选项

### 自定义界面
- 修改 `dash_app.py` 中的布局和样式
- 使用Bootstrap组件美化界面
- 添加新的图表和可视化

### 性能优化
- 使用缓存机制减少重复计算
- 异步处理大图像文件
- 数据库存储替代内存存储

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！
