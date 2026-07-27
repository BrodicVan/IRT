import random, torch, json, os, subprocess

import numpy as np
import pandas as pd

from icecream import ic
from torch.utils.data import Dataset
from typing import Optional
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr


class Sampler:
    """
    数据集采样器，返回原数据集下标的子集。
    """
    def random_sample(self, number_item, total_count, random_seed=42):
        """
        随机挑选测试集
        """
        
        random.seed(random_seed)
        idxs = range(total_count)

        
        seen_items = random.sample(idxs, number_item)
        item_weights = np.ones(number_item)/number_item
            
        # Determine the unseen items by finding all item indices that are not in the seen items list.
        unseen_items = np.setdiff1d(idxs, seen_items).tolist()

        return item_weights, seen_items, unseen_items

    def k_means(self, number_item,inputs,random_seed=42):
        """
        K-Means聚类算法，返回离簇中心最近的样本点，可用于题目的答题分数、IRT参数等
        参数：
        inputs: (total_count * feature_dim)
        """
        total_count = inputs.shape[0]
        kmeans = KMeans(n_clusters=number_item, random_state=random_seed)
        kmeans.fit(inputs)
        distances = kmeans.transform(inputs)  
        seen_items = [np.argmin(distances[:, i]).item() for i in range(number_item)]
        unseen_items = np.setdiff1d(range(total_count), seen_items).tolist()

        labels = kmeans.labels_
        cluster_sizes = np.array([np.sum(labels == i) for i in range(number_item)])
        item_weights = (cluster_sizes / total_count).tolist()
        return item_weights,seen_items,unseen_items

class IRTDataGenerator:
    """
    IRT数据生成器，用于批量生成IRT参数纠正器的IRT相关数据。
    """
    def __init__(self,responses_jsonl_path:str,random_seed=42):
        # 对responses_jsonl_path中的json行进行100轮取5个的抽样
        self.lines = open(responses_jsonl_path,'r',encoding='utf-8').readlines()
        self.model_count = len(self.lines)

    def generate(self, combination_count:int, anchor_count:int, 
                 output_dir:str, 
                 py_irt_params:Optional[dict]=None, random_seed=42):
        """
        选取多种模型组合进行拟合得到IRT参数数据
        """
        print(f"开始生成{combination_count}轮{anchor_count}个模型组合的IRT参数数据")
        combination_set = set()
        idxs = list(range(self.model_count))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        irt_input_path = os.path.join(output_dir,'sampled_responses.jsonl')
        irt_output_dir = os.path.join(output_dir,'fit_output')
        training_data_path = os.path.join(output_dir,'irt_inputs.jsonl')
        

        if py_irt_params is None:
            py_irt_params = {}

        irt_model_type = py_irt_params.get('model_type','2pl')
        py_irt_params['dims'] = py_irt_params.get('dims',10)
        py_irt_params['device'] = py_irt_params.get('device','cuda' if torch.cuda.is_available() else 'cpu')
        

        while len(combination_set) < combination_count:
            sampled_idxs = tuple(sorted(random.sample(idxs,anchor_count)))
            if sampled_idxs in combination_set:
                continue
            combination_set.add(sampled_idxs)
            with open(irt_input_path,'w',encoding='utf-8') as f:
                f.write(''.join([self.lines[i] for i in sampled_idxs]))
            cmd = ['py-irt','train',irt_model_type,irt_input_path,irt_output_dir]
            for key in py_irt_params:
                cmd.extend([f'--{key}',str(py_irt_params[key])])
            subprocess.run(cmd)
            with open(os.path.join(irt_output_dir,'best_parameters.json'),'r',encoding='utf-8') as f:
                content = f.read()
            with open(training_data_path,'a',encoding='utf-8') as f:
                f.write(content.strip()+'\n')
            print(f"当前已生成{len(combination_set)}轮{anchor_count}个模型组合的IRT参数数据")


class Evaluator:
    """
    压缩后数据集评估器
    """
    def __init__(self,responses_df:pd.DataFrame):
        self.items = list(responses_df.columns[1:]) # 题目名称
        self.models = responses_df['source'].tolist() # 模型名称
        self.resposnes_df = responses_df # 原始作答DataFrame


        self.responses_matrix = responses_df[self.items].values # 作答矩阵
        self.scores = self.responses_matrix.mean(axis=1) # 每个模型在全集的平均平均分数
        
        self.sorted_idxs = self.scores.argsort().tolist() # 每个模型在全集的平均分排名

        self.item_count, self.model_count = len(self.items), len(self.models)

        # print(self.sorted_idxs)

    def sort_idx(self,reduced_items:list[int]):
        """
        计算模型在reduced_items上的平均分数并返回排名
        """
        reduced_matrix = self.responses_matrix[:,reduced_items]
        sorted_idxs_new = reduced_matrix.mean(axis=1).argsort().tolist()
        return sorted_idxs_new
    
    def spearman(self,reduced_items:list[int]):
        """
        计算模型的reduced_items排名与全集排名的Spearman相关系数
        """
        return spearmanr(self.sorted_idxs,self.sort_idx(reduced_items))

    def pearman(self,reduced_items:list[int]):
        """
        计算模型的reduced_items排名与全集排名的Pearman相关系数
        """
        return  pearsonr(self.sorted_idxs,self.sort_idx(reduced_items))

    def pairwise_consistence(self,reduced_items:list[int],delta=2):
        """
        计算排名发生变化的模型对数比例，delta表示只统计全集分数相差<delta%的模型对
        """
        
        # 找出全集分数小于delta%的所有模型对
        delta = delta / 100
        model_pairs = []
        for i in range(0,self.model_count-1):
            score_i = self.scores[self.sorted_idxs[i]]
            for j in range(i+1,self.model_count):
                
                if (delta_score:=(self.scores[self.sorted_idxs[j]]-score_i).item())<delta:
                    model_pairs.append((self.sorted_idxs[i],self.sorted_idxs[j]))
                else:
                    break

        # 计算压缩数据集的分数、大小顺序、模型排名
        reduced_scores =  self.responses_matrix[:,reduced_items].mean(axis=1)
        reduced_sorted_idxs = reduced_scores.argsort()
        reduced_sorted_idxs = self.sort_idx(reduced_scores)
        reduced_rank = reduced_sorted_idxs.argmax()

        # 统计排名发生变化的模型对数比例
        count = 0
        for i,j in model_pairs:
            if reduced_rank[i] > reduced_rank[j]:
                count += 1
    
        return count/len(model_pairs)

class MyDataset(Dataset):
    """
    IRT参数纠正器数据，用于生成残差模型的IRT部分训练数据，目前只有离线模式。
    """
    def __init__(self,train_IRT_params:Optional[dict]=None, target_item_params:Optional[dict]=None,
                 item_embeddings:Optional[torch.Tensor]=None, archor_count:int=5,
                 max_sample_count:int=100000,random_seed:int=42):

        if train_IRT_params is None:
            raise ValueError('IRTDataloader: 离线生成模式下，需要传入已拟合好的训练IRT参数（题目参数+模型能力）。')

        if target_item_params is None:
            raise ValueError('IRTDataloader: 需要传入目标题目参数。')

        self.train_IRT_params = train_IRT_params
        self.target_item_params = target_item_params
        self.item_embeddings = item_embeddings
        # TODO: 确定train_IRT_params的格式，从而确定长度

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        return {
            'user_id': row['user_id'],
            'item_id': row['item_id'],
            'response': row['response']
        }

    def get_anchor_items(self):
        np.random.seed(self.random_seed)
        anchor_idxs = np.random.choice(len(self.train_df), self.archor_count, replace=False)
        return self.train_df.iloc[anchor_idxs]

     
    
if __name__=='__main__':
    # sampler = Sampler()
    # number_item = 10
    # inputs = np.random.rand(20, 5)
    # print(sampler.random_sample(number_item,inputs.shape[0])[0])
    # print(sampler.k_means(number_item, inputs)[0])

    # responses_df = pd.read_csv('./resposne_data/processed_data/global_selected_400_test.csv')
    # evaluator = Evaluator(responses_df[:5])
    # evaluator.pairwise_consistence(range(400))
    
    irt_data_generator = IRTDataGenerator('./global_selected_400_train.jsonl')
    irt_data_generator.generate(100,5,'20260727')
    
