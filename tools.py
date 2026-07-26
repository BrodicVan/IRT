import random

import numpy as np
import pandas as pd
from icecream import ic

from sklearn.cluster import KMeans
from scipy.stats import spearmanr


class Sampler:
    
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


class Evaluator:
    def __init__(self,responses_df:pd.DataFrame):
        self.items = list(responses_df.columns[1:]) # 题目名称
        self.models = responses_df['source'].tolist() # 模型名称
        self.resposnes_df = responses_df # 原始作答DataFrame


        self.responses_matrix = responses_df[self.items].values # 作答矩阵
        self.scores = self.responses_matrix.mean(axis=1) # 每个模型在全集的平均平均分数
        
        self.sorted_idxs = self.scores.argsort().tolist() # 每个模型在全集的平均分排名

        self.items_count, self.models_count = len(self.items), len(self.models)

        # print(self.sorted_idxs)


    def sort_idx(self,reduced_items:list[int]):
        """
        计算模型在reduced_items上的平均分数并返回排名
        """
        reduced_matrix = self.responses_matrix[:,reduced_items]
        sorted_idxs_new = reduced_matrix.mean(axis=1).argsort().tolist()
        return sorted_idxs_new
    
    def spearman(self,reduced_items=list[int]):
        """
        计算模型的reduced_items排名与全集排名的Spearman相关系数
        """
        return spearmanr(self.sorted_idxs,self.sort_idx(reduced_items))

    def pairwise_consistence(self,reduced_items=list[int],delta=2):
        """
        计算排名发生变化的模型对数比例，delta表示只统计全集分数相差<delta%的模型对
        """
        
        # 找出全集分数小于delta%的所有模型对
        delta = delta / 100
        
        ic(self.scores)
        ic(self.sorted_idxs)

        model_pairs = []
        
        for i in range(0,self.models_count-1):
            score_i = self.scores[self.sorted_idxs[i]]
            for j in range(i+1,self.models_count):
                
                if (delta_score:=(self.scores[self.sorted_idxs[j]]-score_i).item())<delta:
                    model_pairs.append((self.sorted_idxs[i],self.sorted_idxs[j]))
                else:
                    break
        
        
        # TODO: 检查排名是否变化
        for i,j in model_pairs:
            pass
            



        # print(delta_pairs)
        # print(delta_pairs)
        
            
        
    
if __name__=='__main__':
    # sampler = Sampler()
    # number_item = 10
    # inputs = np.random.rand(20, 5)
    # print(sampler.random_sample(number_item,inputs.shape[0])[0])
    # print(sampler.k_means(number_item, inputs)[0])

    responses_df = pd.read_csv('./resposne_data/processed_data/global_selected_400_test.csv')
    evaluator = Evaluator(responses_df[:5])
    evaluator.pairwise_consistence(range(400))
    
