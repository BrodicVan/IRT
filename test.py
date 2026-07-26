import requests
import pandas as pd

# 访问hf-mirror.com的模型
# url = "https://hf-mirror.com/pansophic/rocket-3B"
# response = requests.get(url)
# print(response.status_code)
# print(response.text)

# 读取CSV文件
# df = pd.read_csv('global_selected_400_scores.csv')
# print(df.head())
# print(df.shape)

# 创建列表存储可访问的模型
# accessible_models = []


# # 便利df中的df['source']列
# for index, row in df.iterrows():
#     source = row['source']
#     url = f"https://hf-mirror.com/{source}"
#     print(index,url)
#     response = requests.get(url)
#     if response.status_code != 200:
#         print(source, response.status_code)
#     else:
#         accessible_models.append(source)


# # 将可访问的模型保存到文件
# with open('accessible_models.txt', 'w', encoding='utf-8') as f:
#     for model in accessible_models:
#         f.write(model + '\n')

# print(f"共找到 {len(accessible_models)} 个可访问模型，已保存到 accessible_models.txt")

# # 读取可访问的模型列表
# with open('accessible_models.txt', 'r', encoding='utf-8') as f:
#     accessible_models = set(line.strip() for line in f.readlines())

# # 遍历df中的source列，记录不可访问的模型
# inaccessible_models = []
# for index, row in df.iterrows():
#     source = row['source']
#     if source not in accessible_models:
#         inaccessible_models.append(source)

# # 将不可访问的模型保存到文件
# with open('inaccessible_models.txt', 'w', encoding='utf-8') as f:
#     for model in inaccessible_models:
#         f.write(model + '\n')

# print(f"共找到 {len(inaccessible_models)} 个不可访问模型，已保存到 inaccessible_models.txt")

# 读取CSV文件并展示前n项
# n = 10  # 设置要展示的行数
# df = pd.read_csv('global_selected_400_data_wide.csv')
# sources = set()

# print(df.columns)

# for column in df.columns:
#     source = column.split('_')[0]
#     sources.add(source)

# print(sources)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 构造示例数据（你可以替换为 pd.read_csv 读入）
data = {
    'Model': ['GPT-4', 'Claude-3', 'Llama-3', 'Gemma'],
    'Q1': [1, 1, 0, 1],
    'Q2': [0, 1, 1, 0],
    'Q3': [1, 1, 1, 1],
    'Q4': [1, 0, 1, 0],
    'Q5': [0, 1, 0, 1]
}
df = pd.DataFrame(data).set_index('Model')
print("原始作答矩阵（1=正确，0=错误）：")
print(df)

# 2. 计算每个模型的正确率（均值）
df['Accuracy'] = df.mean(axis=1) * 100  # 转为百分比

# 3. 按照正确率降序排序
df_sorted = df.sort_values('Accuracy', ascending=False)
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

# 4. 输出排名结果
print("\n" + "="*40)
print("模型排名（按正确率）:")
print(df_sorted[['Accuracy', 'Rank']].round(2))

# 5. 可视化（柱状图）
plt.figure(figsize=(8, 4))
plt.bar(df_sorted.index, df_sorted['Accuracy'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Model Ranking by Answer Accuracy')
plt.ylim(0, 100)
for i, v in enumerate(df_sorted['Accuracy']):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.show()