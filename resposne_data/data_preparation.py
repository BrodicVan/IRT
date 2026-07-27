import pandas as pd
import json, subprocess

origins = ['hellaswag', 'arc', 'gsm8k', 'truthfulqa', 'mmlu', 'winogrande']
def get_origins():
    df = pd.read_csv('./raw_data/global_selected_400_data_wide_filtered.csv')
    # 读取列名，找出公共前缀，prefix__xxx，忽略'source'
    columns = df.columns.tolist()
    origins_set = set()
    for col in columns:
        prefix = col.split('__')[0]
        if prefix != 'source':
            origins_set.add(prefix)
    print(origins_set)

def split_csv_data(train_origins:list[str],test_origins:list[str],train_models:list[str],test_models:list[str]):
    """
    划分csv数据为训练集和测试集，训练集为训练模型在训练集上的作答情况，测试集类推
    """
    # 训练/测试来源必须是origins的子集，且train/test两者并不重叠
    assert set(train_origins).issubset(origins) and set(test_origins).issubset(origins) and set(train_origins).isdisjoint(set(test_origins))

    # 训练模型和测试模型不重叠
    assert set(train_models).isdisjoint(set(test_models))

    # 读取/mnt/fanzhaoji/IRT/general_data/resposne_data/raw_data/global_selected_400_data_wide_filtered.csv并进行划分
    df = pd.read_csv('./raw_data/global_selected_400_data_wide_filtered.csv')

    # 筛选出列名包含train/test_origins的列
    train_cols = ['source'] + [col for col in df.columns if any(prefix in col.split('__')[0] for prefix in train_origins)]
    test_cols = ['source'] + [col for col in df.columns if any(prefix in col.split('__')[0] for prefix in test_origins)]

    # 根据origins划分出特定df列
    train_df = df[train_cols]
    test_df = df[test_cols]

    # 根据models划分出特定df行
    train_df = train_df[train_df['source'].isin(train_models)]
    test_df = test_df[test_df['source'].isin(test_models)]
    
    # 保存训练集和测试集
    train_df.to_csv('./processed_data/global_selected_400_train.csv',index=False)
    test_df.to_csv('./processed_data/global_selected_400_test.csv',index=False)
    
def check_csv_info(csv_path):
    df = pd.read_csv(csv_path)
    print(df.info())
    print(df.head())

def sample_and_save_csv(input_path, output_path, n_rows=20, n_cols=20):
    """
    读取 CSV 文件的前 n 行和 n 列，并另存为新文件
    """
    df = pd.read_csv(input_path)
    sample_df = df.iloc[:n_rows, :n_cols]
    sample_df.to_csv(output_path, index=False)
    print(f"已保存前{n_rows}行和{n_cols}列到{output_path}")

def csv2jsonl(csv_path,type='wide'):
    """
    将csv转为py-irt使用的jsonl格式
    type: wide or narrow
    """
    df = pd.read_csv(csv_path)
    if type=='wide':
        # 保存jsonl文件
        with open(csv_path.replace('.csv','.jsonl'),'w') as f:
            for idx,row_series in df.iterrows():
                
                
                # Series转json
                data = {
                    "subject_id": row_series.iloc[0], 
                    "responses": row_series[1:].to_dict()
                }
                f.write(json.dumps(data,ensure_ascii=False)+'\n')
    elif type=='narrow':
        # narrow 格式：每行一个观测记录
        with open(csv_path.replace('.csv', '_narrow.jsonl'), 'w') as f:
            for _, row_series in df.iterrows():
                subject_id = row_series.iloc[0]
                for col in df.columns[1:]:
                    if pd.notna(row_series[col]):
                        data = {
                            "subject_id": subject_id,
                            "item_id": col,
                            "response": row_series[col]
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    else:
        raise ValueError('type must be "wide" or "narrow"')

    
    

def fit_irt(jsonl_path):
    """
    使用py-irt拟合IRT参数
    """
    # subprocess.run()
    pass

if __name__=='__main__':
    # 获取所有数据来源（jiqi
    # get_origins()

    # 划分训练/测试数据
    # with open('/mnt/fanzhaoji/IRT/general_data/resposne_data/model_split.json','r') as f:
    #     model_split = json.load(f)
    # train_models = model_split['train_models']
    # test_models = model_split['test_models']
    # train_origins = ['mmlu']
    # test_origins = list(set(origins) - set(train_origins))
    # split_csv_data(train_origins=train_origins,test_origins=test_origins,
    #                train_models=train_models,test_models=test_models)

    # check_csv_info('/mnt/fanzhaoji/IRT/general_data/resposne_data/processed_data/global_selected_400_train.csv')
    # sample_and_save_csv('./processed_data/global_selected_400_test.csv','./processed_data/test.csv')
    csv2jsonl('./processed_data/test.csv','narrow')


    
