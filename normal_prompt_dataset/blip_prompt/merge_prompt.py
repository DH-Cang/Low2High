import json
import os

# 定义要合并的 JSON 文件路径
json_files = [f"./training_{i}.json" for i in range(6)]

# 初始化一个空字典来存储合并后的数据
merged_data = {}

# 读取每个 JSON 文件并合并到 merged_data 字典中
for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        merged_data.update(data)

# 将合并后的数据写入新的 JSON 文件
output_file = 'merged_prompt.json'
with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"合并后的 JSON 数据已保存到 {output_file}")