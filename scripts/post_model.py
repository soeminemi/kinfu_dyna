import requests
import sys

import consts

body_measure_path = sys.argv[1]  # 从命令行参数获取文件路径
name = sys.argv[2]  # 从命令行参数获取姓名

with open(body_measure_path, 'r') as f:  # 使用命令行参数中的路径
    body_model = f.read()

height = '170'
gender = 'female'
custom = 'body_measure'
model_name = f"{gender}-{name}-{custom}-{height}.ply"
input = {
    'name': consts.name,
    'passwd': consts.passwd,
    "body_model": body_model,
    'model_name': model_name,
}

# 定义API端点
api_endpoint = f'http://{consts.measure_front_host}/push_3d_model'
# 发送POST请求
response = requests.post(api_endpoint, json=input, timeout=10000)
print(response)
rec = response.json()
# print(response.status_code, rec)
url = f'http://{consts.measure_front_host}/measure?model_name={model_name}'
print(url)
# redirect(url)
