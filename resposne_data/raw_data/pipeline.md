# 数据处理pipeline

本文件用于完成一个完整的数据整理流程。按 6 个 benchmark 全集总体正确率抽样：

1. 读取 6 个 `benchmark-data/*-preproc.rds`。
2. 找出 6 个 benchmark 都存在的共同 LLM。
3. 在 6 个 benchmark 的全部题目上累计总体正确率。
4. 按总体正确率从低到高等间距抽样，选出 400 个覆盖 global 能力范围的 LLM。
5. 导出这 400 个 LLM 在每个 benchmark 上的逐题数据。
6. 将 6 个 benchmark 横向合并成一个 global 数据集。
7. 核查选出的 400 个模型是否在所有题目上都有回答。
8. 核查这 400 个模型在总题目上的准确率分布是否平滑。
9. 所有结果都导出为普通 CSV，不再使用 RDS。

## 文件说明

本数据集包含 400 个 selected LLM 在 6 个 benchmark 共 22,058 道题上的逐题正确性结果。

1. `global_selected_400_data_wide.csv`  
   核心 response matrix。行是模型，第一列 `source` 为模型名；列是题目，列名为 `global_item`；单元格为 1/0，表示 correct/incorrect。

2. `global_items.csv`  
   题目 metadata。每行对应一道题，用 `global_item` 与 response matrix 的列名连接；`benchmark` 表示题目来源，`local_item` 是原 benchmark 内的题目 ID。

3. `global_selected_400_scores.csv`  
   模型汇总分数。每行对应一个模型，用 `source` 与 response matrix 连接；`accuracy` 是该模型在全部 22,058 道题上的总体正确率。

其中，`global_selected_400_scores.csv` 可由 response matrix 重新计算得到；`global_items.csv` 提供题目层面的补充信息(题目Prompt等)。


## 总体正确率定义

对每个共同模型，先在 6 个 benchmark 上分别统计逐题结果，再累计为 global 分数：accuracy=回答正确题目数/总题数


## 排序和等间距抽样代码

```r
select_evenly <- function(df, n) {
  # 按 global accuracy 升序排序，同分按 source 排序，确保低 global 能力模型在前
  df <- df[order(df$accuracy, df$source), , drop = FALSE]
  rownames(df) <- NULL

  # 如果要选择的数量 >= 总模型数，则直接返回全部，并标记 selection_rank
  if (n >= nrow(df)) {
    df$selection_rank <- seq_len(nrow(df))
    return(df)
  }

  # 计算均匀选择的位置索引
  # seq(1, nrow(df), length.out = n) 生成 n 个均匀间隔的浮点数位置
  # round() 转为整数索引
  positions <- round(seq(1, nrow(df), length.out = n))

  # 处理因 round() 可能导致的重复索引
  used <- integer()
  repaired <- integer()
  for (pos in positions) {
    if (!(pos %in% used)) {
      used <- c(used, pos)
      repaired <- c(repaired, pos)
      next
    }

    # 如果索引重复，向左右搜索最近未用的索引
    radius <- 1
    repeat {
      candidates <- c(pos - radius, pos + radius)
      candidates <- candidates[candidates >= 1 & candidates <= nrow(df)]
      candidates <- candidates[!(candidates %in% used)]
      if (length(candidates) > 0) {
        used <- c(used, candidates[[1]])
        repaired <- c(repaired, candidates[[1]])
        break
      }
      radius <- radius + 1
    }
  }

  # 根据修复后的索引选出模型，并按 selection_rank 标记顺序
  out <- df[sort(repaired), , drop = FALSE]
  out$selection_rank <- seq_len(nrow(out))
  out
}
```

## 处理后

| benchmark | 题目数 | 模型数 | 模型-题目单元 |
|---|---:|---:|---:|
| arc | 844 | 400 | 337,600 |
| gsm8k | 1,306 | 400 | 522,400 |
| hellaswag | 5,711 | 400 | 2,284,400 |
| mmlu | 12,508 | 400 | 5,003,200 |
| truthfulqa | 644 | 400 | 257,600 |
| winogrande | 1,045 | 400 | 418,000 |
| GLOBAL | 22,058 | 400 | 8,823,200 |

## 回答完整性核查

结论：400 个 selected 模型在 6 个 benchmark 的全部 22,058 道题上都有回答。


## 全局准确率分布核查

400 个模型是在 6 个 benchmark 全集的 global accuracy 上等间距抽样得到的。它们在 22,058 道 global 题目上的准确率统计如下：

| 指标 | 数值 |
|---|---:|
| min | 0.1779 |
| median | 0.6695 |
| max | 0.8204 |
| mean | 0.6272 |
| population sd | 0.1392 |

按 global accuracy 从低到高排序后，相邻模型准确率 gap：

| 指标 | 数值 |
|---|---:|
| mean gap | 0.00161 |
| median gap | 0.00068 |
| max gap | 0.01401 |

这说明 selected 400 在 global 总题目上的准确率排序整体比较平滑。

等宽 10 个准确率区间的模型数：

| bin | accuracy_low | accuracy_high | model_count |
|---:|---:|---:|---:|
| 1 | 0.1779 | 0.2421 | 11 |
| 2 | 0.2421 | 0.3064 | 15 |
| 3 | 0.3064 | 0.3707 | 9 |
| 4 | 0.3707 | 0.4349 | 12 |
| 5 | 0.4349 | 0.4992 | 11 |
| 6 | 0.4992 | 0.5634 | 24 |
| 7 | 0.5634 | 0.6277 | 45 |
| 8 | 0.6277 | 0.6919 | 116 |
| 9 | 0.6919 | 0.7562 | 141 |
| 10 | 0.7562 | 0.8204 | 16 |


