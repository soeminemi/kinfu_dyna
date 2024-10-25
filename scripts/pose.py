import asyncio
import websockets
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import math
import json

# 加载YOLO模型
model = YOLO('yolov8n-pose.pt')

async def calculate_arm_angles(image):
    # 运行YOLO模型进行姿态估计
    results = model(image)
    
    # 获取关键点
    keypoints = results[0].keypoints.xy[0].cpu().numpy()
    
    # 提取左右肩膀和手腕的坐标
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    # 关键点序号对应的关节点位：
    # 0: 鼻子
    # 1: 左眼
    # 2: 右眼
    # 3: 左耳
    # 4: 右耳
    # 5: 左肩
    # 6: 右肩
    # 7: 左肘
    # 8: 右肘
    # 9: 左手腕
    # 10: 右手腕
    # 11: 左髋
    # 12: 右髋
    # 13: 左膝
    # 14: 右膝
    # 15: 左脚踝
    # 16: 右脚踝
    
    # 计算手臂向量
    left_arm_vector = left_wrist - left_shoulder
    right_arm_vector = right_wrist - right_shoulder
    
    # 计算手臂与水平线的夹角
    left_angle = math.degrees(math.atan2(left_arm_vector[1], left_arm_vector[0]))
    right_angle = math.degrees(math.atan2(right_arm_vector[1], right_arm_vector[0]))
    
    # 调整角度范围到0-180度
    left_angle = (left_angle + 360) % 360
    right_angle = (right_angle + 360) % 360
    if left_angle > 180:
        left_angle = 360 - left_angle
    if right_angle > 180:
        right_angle = 360 - right_angle
    # 绘制关节图
    image_with_keypoints = image.copy()
    
    # 定义关节连接
    connections = [
        (5, 7), (7, 9),  # 左臂
        (6, 8), (8, 10),  # 右臂
        (5, 6), (5, 11), (6, 12),  # 躯干
        (11, 13), (13, 15),  # 左腿
        (12, 14), (14, 16)  # 右腿
    ]
    
    # 绘制关节点
    for point in keypoints:
        cv2.circle(image_with_keypoints, tuple(point.astype(int)), 5, (0, 255, 0), -1)
    
    # 绘制连接线
    for connection in connections:
        start_point = tuple(keypoints[connection[0]].astype(int))
        end_point = tuple(keypoints[connection[1]].astype(int))
        cv2.line(image_with_keypoints, start_point, end_point, (255, 0, 0), 2)
    
    # 保存结果图像
    cv2.imwrite('result.jpg', image_with_keypoints)
    return left_angle, right_angle

async def handle_websocket(websocket, path):
    async for message in websocket:
        # 解码Base64图像
        image_data = base64.b64decode(message)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 计算手臂角度
        left_angle, right_angle = await calculate_arm_angles(image)
        
        # 构造响应
        response = json.dumps({
            "left_angle": left_angle,
            "right_angle": right_angle
        })
        
        # 发送响应
        await websocket.send(response)

async def main():
    server = await websockets.serve(handle_websocket, "localhost", 8765)
    print("WebSocket服务器已启动，监听端口8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())


def save_parameters_to_mat(self, output_dir):
        """
        将模型参数保存为文本格式，但文件后缀为.mat，并在文件开头添加行数和列数信息
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        def save_array_with_header(filename, array, fmt):
            # 获取数组的形状
            shape = array.shape
            # 创建头部信息
            header = f"{shape[0]} {shape[1]}" if len(shape) > 1 else f"{shape[0]} 1"
            # 保存数组，包括头部信息
            np.savetxt(filename, array, fmt=fmt, header=header, comments='')

        # 保存各个参数
        save_array_with_header(os.path.join(output_dir, 'J_regressor.mat'), self.J_regressor.cpu().numpy(), fmt='%.6f')
        save_array_with_header(os.path.join(output_dir, 'weights.mat'), self.weights.cpu().numpy(), fmt='%.6f')
        save_array_with_header(os.path.join(output_dir, 'posedirs.mat'), self.posedirs.cpu().numpy(), fmt='%.6f')
        save_array_with_header(os.path.join(output_dir, 'v_template.mat'), self.v_template.cpu().numpy(), fmt='%.6f')
        save_array_with_header(os.path.join(output_dir, 'shapedirs.mat'), self.shapedirs.cpu().numpy(), fmt='%.6f')
        save_array_with_header(os.path.join(output_dir, 'faces.mat'), self.faces.cpu().numpy(), fmt='%d')
        save_array_with_header(os.path.join(output_dir, 'kintree_table.mat'), self.kintree_table.cpu().numpy(), fmt='%d')
        save_array_with_header(os.path.join(output_dir, 'parent.mat'), self.parent.cpu().numpy(), fmt='%d')

        print(f"所有参数已保存到 {output_dir} 目录下的.mat文件中（文本格式，包含行列数信息）")