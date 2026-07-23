import requests
from bs4 import BeautifulSoup
import re
import json
import time
import pandas as pd
import random

def parse_file_size(size_str):
    size_str = size_str.strip().lower()
    if 'kb' in size_str:
        return float(size_str.replace('kb', '')) / 1024
    elif 'mb' in size_str:
        return float(size_str.replace('mb', ''))
    elif 'gb' in size_str:
        return float(size_str.replace('gb', '')) * 1024
    return 0.0

def get_model_size(model_name):
    url = f"https://hf-mirror.com/{model_name}/tree/main?not-for-all-audiences=true"
    try:
        response = requests.get(url, timeout=10)
        while response.status_code == 429:
            print(f"{model_name}: Status code {response.status_code}")
            time.sleep(30)
            response = requests.get(url, timeout=10)
        

        soup = BeautifulSoup(response.text, 'html.parser')
        total_size_mb = 0
        for target in soup.find_all('a', href=re.compile("(.*\\.(safetensors))\\?download=true")):
            span = target.find('span')
            if span:
                print('safetensors found')
                total_size_mb += parse_file_size(span.text)

        if total_size_mb == 0:
            
            for target in soup.find_all('a', href=re.compile("(.*\\.(bin))\\?download=true")):
                span = target.find('span')
                if span:
                    print('bin found')
                    total_size_mb += parse_file_size(span.text)
        

        if total_size_mb == 0:
            return 9999
        else:
            return total_size_mb / 1024 
    
    except Exception as e:
        print(f"Error accessing {model_name}: {e}")
        print(response.status_code)
        return 9999

def init_size():
    with open('model_data/accessible_models.txt', 'r', encoding='utf-8') as f:
        model_names = [line.strip() for line in f if line.strip()]
    
    results = []
    for i, model in enumerate(model_names):
        print(f"Processing {i+1}/{len(model_names)}: {model}")
        model_size = get_model_size(model)
        results.append(
            {
                'model': model,
                'model_size': model_size
            }
        )
            
        results.sort(key=lambda x: x['model_size'], reverse=True)
        
        
        
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        for res in results:
            if res['model_size'] is not None:
                print(f"{res['model']:50} | {res['model_size']:6.2f} GB")
            else:
                print(f"{res['model']:50} | {'Failed':>10} | Unknown")
    
    with open('model_data/accessible_model_sizes.jsonl', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to model_data/accessible_model_sizes.jsonl")

def fix():
    with open('model_data/accessible_model_sizes.jsonl', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    failed_models = [item for item in results if item['model_size'] == 9999]
    print(f"Found {len(failed_models)} models with size 9999 to fix")
    
    for i, item in enumerate(failed_models):
        model_name = item['model']
        print(f"Fixing {i+1}/{len(failed_models)}: {model_name}")
        new_size = get_model_size(model_name)
        
        if new_size != 9999:
            item['model_size'] = new_size
            print(f"  Updated: {new_size:.2f} GB")
        else:
            print(f"  Still failed")
        
        results.sort(key=lambda x: x['model_size'], reverse=True)
        
        with open('model_data/accessible_model_sizes.jsonl', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to model_data/accessible_model_sizes.jsonl")
    
    print("\n" + "="*60)
    print("Fix Summary:")
    print("="*60)
    remaining_failed = sum(1 for item in results if item['model_size'] == 9999)
    print(f"Total models: {len(results)}")
    print(f"Still failed: {remaining_failed}")


def rank_accessible_models():
    
    # 读取模型大小数据
    with open('./model_data/filtered_models.jsonl', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 使用pandas读取raw_data/global_selected_400_scores.csv获取模型正确率信息
    model_accuracy = {}
    try:
        df = pd.read_csv('raw_data/global_selected_400_scores.csv')
        # 假设csv包含'model'和'accuracy'列
        if 'model' in df.columns and 'accuracy' in df.columns:
            # 去除空白字符并转换为字典
            df['model'] = df['model'].str.strip()
            model_accuracy = df.set_index('model')['accuracy'].to_dict()
        else:
            print("Warning: global_items.csv缺少'model'或'accuracy'列")
    except FileNotFoundError:
        print("Warning: raw_data/global_selected_400_scores.csv not found, will use only size for sorting")
    except Exception as e:
        print(f"Error reading raw_data/global_selected_400_scores.csv: {e}")
        results = json.load(f)
    
    results.sort(key=lambda x: x['model_size'], reverse=True)
    
    # 为每个结果添加正确率信息，并按正确率排序（降序）
    for item in results:
        model_name = item['model']
        item['accuracy'] = model_accuracy.get(model_name, 9999)
    
    # 按正确率降序排序，正确率相同的按模型大小降序排序
    results.sort(key=lambda x: (x['accuracy'], x['model_size']), reverse=True)
    
    # 保存排序后的结果
    with open('model_data/ranked_models.jsonl', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to model_data/ranked_models.jsonl")
    print("\n" + "="*80)
    print("Ranking Summary (sorted by accuracy):")
    print("="*80)
    print(f"{'Model':<50} | {'Size (GB)':>10} | {'Accuracy':>10}")
    print("-"*80)
    for res in results:
        size_str = f"{res['model_size']:6.2f}" if res['model_size'] != 9999 else "Failed"
        acc_str = f"{res['accuracy']:6.2%}" if res['accuracy'] != 9999 else "Unknown"
        print(f"{res['model']:62} | {size_str:>10} | {acc_str:>10}")
    print("="*80)

def filter():
    with open('model_data/accessible_model_sizes.jsonl', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    results = [item for item in results if item['model_size'] < 30 and item['model_size'] != 9999]
    
    for idx,result in enumerate(results):
        print(f"Processing {idx+1}/{len(results)}: {result['model']}")

        url = f"https://hf-mirror.com/{result['model']}/tree/main?not-for-all-audiences=true"
        response = requests.get(url, timeout=10)
        while response.status_code == 429:
            print(f"{result['model']}: Status code {response.status_code}")
            time.sleep(30)
            response = requests.get(url, timeout=10)
        

        soup = BeautifulSoup(response.text, 'html.parser')
        adapter_link = soup.find('a', href=re.compile(".*adapter.*"))
        if adapter_link:
            results.remove(result)
            print(f"{result['model']}: Adapter found, removed")
    
    with open('model_data/filtered_models.jsonl', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def frequency_sample():
    # 读取模型数据
    with open('./model_data/ranked_models.jsonl', 'r', encoding='utf-8') as f:
        results = json.load(f)
     
    
    # 按正确率排序
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    model_count = len(results)
    chunk_size = (model_count+4) // 5

    
    # 计算并打印5分点
    accuracies = [item['accuracy'] for item in results if item['accuracy'] != 9999]
    if accuracies:
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        print("\n" + "="*60)
        print("正确率5分点:")
        print("="*60)
        for q in quantiles:
            value = pd.Series(accuracies).quantile(q)
            print(f"  {int(q*100)}%分点: {value:.4f}")
        print("="*60)
    
    boxes = []
    i = 0
    while i < 4:
        boxes.append(results[i*chunk_size:(i+1)*chunk_size])
        i += 1
    boxes.append(results[i*chunk_size:])
    
    for box in boxes:
        print(len(box))

    balance_count = len(boxes[-1])

    selected_models = []
    for _ in range(balance_count):
        cur_iter = []
        for box in boxes:
            if box:
                selected = random.choice(box)
                cur_iter.append(selected)
                box.remove(selected)
        random.shuffle(cur_iter)
        selected_models.extend(cur_iter)
    
    # 将剩下的模型放进selected_models
    for box in boxes:
        selected_models.extend(box)
    
    # 保存结果
    with open('model_data/frequency_sample_models.jsonl', 'w', encoding='utf-8') as f:
        json.dump(selected_models, f, ensure_ascii=False, indent=2)
    
    print(f"\n总共选择了 {len(selected_models)} 个模型")
    print(f"结果已保存到 model_data/frequency_sample_models.jsonl")
    
if __name__ == '__main__':
    # init_size()
    # fix()
    # filter()
    # rank_accessible_models()
    
    frequency_sample(
        
    )
