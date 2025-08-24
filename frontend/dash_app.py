import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
import base64
import io
from PIL import Image
import json
from datetime import datetime
import numpy as np

# 初始化Dash应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "缺陷检测系统"

# API服务器地址
API_BASE_URL = "http://localhost:8080/api"

# 应用布局
app.layout = dbc.Container([
    # 标题栏
    dbc.Row([
        dbc.Col([
            html.H1("智能缺陷检测系统", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # 主要内容区域
    dbc.Row([
        # 左侧控制面板
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("控制面板"),
                dbc.CardBody([
                    # 图像上传
                    html.H5("图像上传"),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            '拖拽图像文件到此处或 ',
                            html.A('点击选择文件')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-progress', className='mt-2'),
                    
                    html.Hr(),
                    
                    # 检测方法选择
                    html.H5("检测方法"),
                    dcc.Dropdown(
                        id='detection-method',
                        options=[
                            {'label': '边缘检测', 'value': 'edge_detection'},
                            {'label': '阈值分析', 'value': 'threshold_analysis'},
                            {'label': '纹理分析', 'value': 'texture_analysis'},
                            {'label': '颜色聚类', 'value': 'color_clustering'},
                            {'label': '形态学分析', 'value': 'morphological_analysis'},
                            {'label': '综合检测', 'value': 'comprehensive'}
                        ],
                        value='edge_detection',
                        className="mb-3"
                    ),
                    
                    # 检测按钮和进度显示
                    dbc.Button(
                        "开始检测",
                        id="detect-button",
                        color="primary",
                        size="lg",
                        className="w-100 mb-2",
                        disabled=True
                    ),
                    html.Div(id='detection-progress', className='mb-2'),
                    
                    html.Hr(),
                    
                    # 系统状态
                    html.H5("系统状态"),
                    html.Div(id="system-status"),
                    
                    html.Hr(),
                    
                    # 操作按钮
                    dbc.ButtonGroup([
                        dbc.Button("查看历史", id="history-button", color="info", size="sm"),
                        dbc.Button("清除历史", id="clear-button", color="warning", size="sm"),
                        dbc.Button("刷新统计", id="refresh-button", color="success", size="sm")
                    ], className="w-100")
                ])
            ])
        ], width=3),
        
        # 中间图像显示区域
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("图像显示"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(label="原始图像", tab_id="original-tab"),
                        dbc.Tab(label="处理结果", tab_id="processed-tab")
                    ], id="image-tabs", active_tab="original-tab"),
                    
                    html.Div(id="image-display", className="mt-3")
                ])
            ])
        ], width=6),
        
        # 右侧结果面板
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("检测结果"),
                dbc.CardBody([
                    html.Div(id="detection-results")
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("统计信息"),
                dbc.CardBody([
                    html.Div(id="statistics-display")
                ])
            ])
        ], width=3)
    ]),
    
    html.Hr(),
    
    # 底部历史记录和图表
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("检测历史"),
                dbc.CardBody([
                    html.Div(id="history-display")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("趋势图表"),
                dbc.CardBody([
                    dcc.Graph(id="trend-chart")
                ])
            ])
        ], width=6)
    ]),
    
    # 隐藏的存储组件
    dcc.Store(id='uploaded-image-data'),
    dcc.Store(id='detection-result-data'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # 5秒刷新一次
    
], fluid=True)

# 回调函数：显示上传进度
@app.callback(
    Output('upload-progress', 'children'),
    [Input('upload-image', 'contents')],
    prevent_initial_call=True
)
def update_upload_progress(contents):
    if contents is None:
        return html.Div("等待上传图像...", className="text-muted small")
    
    return dbc.Progress(
        value=100, 
        striped=True, 
        animated=True, 
        label="上传完成!",
        color="success",
        className="mb-2"
    )

# 回调函数：处理图像上传
@app.callback(
    [Output('uploaded-image-data', 'data'),
     Output('detect-button', 'disabled', allow_duplicate=True),
     Output('image-display', 'children', allow_duplicate=True)],
    [Input('upload-image', 'contents')],
    [State('image-tabs', 'active_tab')],
    prevent_initial_call=True
)
def handle_image_upload(contents, active_tab):
    if contents is None:
        return None, True, html.Div("请上传图像文件", className="text-center text-muted")
    
    try:
        # 解析上传的图像
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(io.BytesIO(decoded))
        
        # 显示原始图像
        if active_tab == "original-tab":
            image_display = html.Img(
                src=contents,
                style={'width': '100%', 'max-height': '400px', 'object-fit': 'contain'}
            )
        else:
            image_display = html.Div("请先进行检测", className="text-center text-muted")
        
        return contents, False, image_display
        
    except Exception as e:
        return None, True, html.Div(f"图像上传失败: {str(e)}", className="text-danger")

# 回调函数：执行缺陷检测（合并进度显示和检测执行）
@app.callback(
    [Output('detection-result-data', 'data'),
     Output('detection-results', 'children'),
     Output('detection-progress', 'children'),
     Output('detect-button', 'disabled', allow_duplicate=True),
     Output('detect-button', 'color', allow_duplicate=True)],
    [Input('detect-button', 'n_clicks')],
    [State('uploaded-image-data', 'data'),
     State('detection-method', 'value')],
    prevent_initial_call=True,
    background=True,
    running=[
        (Output('detection-progress', 'children'), dbc.Progress(value=0, striped=True, animated=True, label="检测中...", color="info", className="mb-2"), ""),
        (Output('detect-button', 'disabled', allow_duplicate=True), True, False),
        (Output('detect-button', 'color', allow_duplicate=True), "secondary", "primary")
    ]
)
def perform_detection(n_clicks, image_data, method):
    ctx = callback_context
    
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if n_clicks is None or image_data is None:
        return None, html.Div("请先上传图像", className="text-muted"), html.Div("请先上传图像", className="text-muted small"), True, "primary"
    
    try:
        # 发送检测请求
        response = requests.post(f"{API_BASE_URL}/detect", json={
            'image': image_data,
            'method': method
        })
        
        if response.status_code == 200:
            result = response.json()
            
            # 显示检测结果
            if result['status'] == 'success':
                defects_found = result.get('defects_found', 0)
                confidence = result.get('confidence_score', 0.0)
                processing_time = result.get('processing_time', 0.0)
                
                result_display = [
                    dbc.Alert([
                        html.H5("检测完成!", className="alert-heading"),
                        html.P(f"发现缺陷: {defects_found} 个"),
                        html.P(f"置信度: {confidence:.2%}"),
                        html.P(f"处理时间: {processing_time:.3f} 秒")
                    ], color="success" if defects_found == 0 else "warning"),
                ]
                
                # 显示缺陷详情
                if defects_found > 0:
                    defects_table = dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("ID"),
                                html.Th("类型"),
                                html.Th("面积"),
                                html.Th("置信度")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(defect['id']),
                                html.Td(defect['type']),
                                html.Td(f"{defect['area']:.0f}"),
                                html.Td(f"{defect['confidence']:.2%}")
                            ]) for defect in result.get('defects', [])[:5]  # 只显示前5个
                        ])
                    ], striped=True, bordered=True, hover=True, size="sm")
                    
                    result_display.append(html.H6("缺陷详情:"))
                    result_display.append(defects_table)
                
                return result, result_display, None, False, "primary"
            else:
                return result, dbc.Alert(f"检测失败: {result.get('error_message', '未知错误')}", color="danger"), None, False, "primary"
        else:
            return None, dbc.Alert("API请求失败", color="danger"), None, False, "primary"
            
    except Exception as e:
        return None, dbc.Alert(f"检测过程中出现错误: {str(e)}", color="danger"), None, False, "primary"

# 回调函数：更新图像显示
@app.callback(
    Output('image-display', 'children'),
    [Input('image-tabs', 'active_tab'),
     Input('detection-result-data', 'data')],
    [State('uploaded-image-data', 'data')]
)
def update_image_display(active_tab, detection_result, original_image):
    if active_tab == "original-tab":
        if original_image:
            return html.Img(
                src=original_image,
                style={'width': '100%', 'max-height': '400px', 'object-fit': 'contain'}
            )
        else:
            return html.Div("请上传图像文件", className="text-center text-muted")
    
    elif active_tab == "processed-tab":
        if detection_result and 'processed_image' in detection_result:
            return html.Img(
                src=detection_result['processed_image'],
                style={'width': '100%', 'max-height': '400px', 'object-fit': 'contain'}
            )
        else:
            return html.Div("请先进行检测", className="text-center text-muted")

# 回调函数：更新系统状态
@app.callback(
    Output('system-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_system_status(n):
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            status = dbc.Badge("系统正常", color="success", className="mb-2")
        else:
            status = dbc.Badge("系统异常", color="danger", className="mb-2")
    except:
        status = dbc.Badge("连接失败", color="danger", className="mb-2")
    
    return [
        status,
        html.P(f"最后更新: {datetime.now().strftime('%H:%M:%S')}", className="small text-muted")
    ]

# 回调函数：更新统计信息
@app.callback(
    Output('statistics-display', 'children'),
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_statistics(n_clicks, n_intervals):
    try:
        response = requests.get(f"{API_BASE_URL}/statistics")
        if response.status_code == 200:
            stats = response.json()
            
            return [
                html.P([html.Strong("总检测次数: "), f"{stats['total_detections']}"]),
                html.P([html.Strong("成功率: "), f"{stats['success_rate']:.1f}%"]),
                html.P([html.Strong("平均处理时间: "), f"{stats['avg_processing_time']:.3f}s"]),
                html.Hr(),
                html.H6("方法使用统计:"),
                html.Ul([
                    html.Li(f"{method}: {count}次") 
                    for method, count in stats.get('method_usage', {}).items()
                ])
            ]
        else:
            return html.P("无法获取统计信息", className="text-muted")
    except:
        return html.P("连接失败", className="text-danger")

# 回调函数：更新历史记录
@app.callback(
    Output('history-display', 'children'),
    [Input('history-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_history(n_clicks, n_intervals):
    try:
        response = requests.get(f"{API_BASE_URL}/history?limit=10")
        if response.status_code == 200:
            data = response.json()
            history = data.get('history', [])
            
            if not history:
                return html.P("暂无检测历史", className="text-muted")
            
            history_items = []
            for item in history:
                timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                badge_color = "success" if item['defects_found'] == 0 else "warning"
                
                history_items.append(
                    dbc.ListGroupItem([
                        html.Div([
                            html.Strong(f"检测ID: {item['id'][:8]}..."),
                            dbc.Badge(f"{item['defects_found']} 缺陷", color=badge_color, className="ms-2")
                        ]),
                        html.Small([
                            f"方法: {item['method']} | ",
                            f"置信度: {item['confidence_score']:.2%} | ",
                            f"时间: {timestamp.strftime('%H:%M:%S')}"
                        ], className="text-muted")
                    ])
                )
            
            return dbc.ListGroup(history_items)
        else:
            return html.P("无法获取历史记录", className="text-muted")
    except:
        return html.P("连接失败", className="text-danger")

# 回调函数：更新趋势图表
@app.callback(
    Output('trend-chart', 'figure'),
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_trend_chart(n_clicks, n_intervals):
    try:
        response = requests.get(f"{API_BASE_URL}/statistics")
        if response.status_code == 200:
            stats = response.json()
            trend_data = stats.get('defects_trend', [])
            
            if not trend_data:
                return go.Figure().add_annotation(
                    text="暂无数据",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            df = pd.DataFrame(trend_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            
            # 缺陷数量趋势
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['defects_found'],
                mode='lines+markers',
                name='缺陷数量',
                line=dict(color='red')
            ))
            
            # 置信度趋势
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['confidence'],
                mode='lines+markers',
                name='置信度',
                yaxis='y2',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="检测趋势",
                xaxis_title="时间",
                yaxis=dict(title="缺陷数量", side="left"),
                yaxis2=dict(title="置信度", side="right", overlaying="y"),
                height=300,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        else:
            return go.Figure().add_annotation(
                text="无法获取数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    except:
        return go.Figure().add_annotation(
            text="连接失败",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

# 回调函数：清除历史记录
@app.callback(
    Output('clear-button', 'children'),
    [Input('clear-button', 'n_clicks')]
)
def clear_history(n_clicks):
    if n_clicks:
        try:
            response = requests.post(f"{API_BASE_URL}/clear_history")
            if response.status_code == 200:
                return "已清除"
            else:
                return "清除失败"
        except:
            return "连接失败"
    return "清除历史"

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Defect Detection Dashboard')
    parser.add_argument('--port', type=int, default=12000, help='Port to run the dashboard on')
    parser.add_argument('--api-port', type=int, default=8080, help='Port of the API server')
    args = parser.parse_args()
    
    # 更新API基础URL
    API_BASE_URL = f"http://localhost:{args.api_port}/api"
    
    print("Starting Defect Detection Dashboard...")
    print(f"Dashboard will be available at: http://localhost:{args.port}")
    print(f"API server at: {API_BASE_URL}")
    app.run(host='0.0.0.0', port=args.port, debug=True)
