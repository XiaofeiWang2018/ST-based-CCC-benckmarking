"""

-----------
mouse_kidney_samples文件夹下有5个mouse_kidney sample,mouse_kidney_data为cell*genes的矩阵，
mouse_kidney_meta储存有细胞坐标和标注，根据不同CCC method需求处理成需要的输入，有可能会用到lr_pairs.csv和
scdata.h5ad。以上数据可满足大部分CCC输入要求，如果碰到问题在群里@我。

不同的CCC方法的输出大多都可以整理成cell type-LR pairs，和single-cell-LR pairs中的一种或者两种，

对于cell type-LR pairs，参考demo1_cell_type/pred_edges.csv
对于这一类CCC的工具的输出：整理为标准化边 (sender_ct, receiver_ct, ligand, receptor)：
如果工具已经输出了筛选后的显著相互作用，你可以直接把这些边写入 pred_edges.csv；如果工具输出未筛选
的全量结果，则先按统一规则筛选（例如 p-value < 0.05，或按 score/rank 取 top 10%），把筛选后的
边合并去重生成 pred_edges.csv。随后基于 pred_edges.csv 在每个 (dataset_id, sample_id) 内统
计每条边在多少个工具中出现，并把出现次数 ≥ 3（或你设定的阈值）的边写入 shared_edges.csv。

对于single-cell-LR pairs，参考demo2_single_cell/edges_long.csv
同理对于这一类CCC的工具的输出：整理为标准化边 (sender_ID, receiver_ID, ligand, receptor)，
根据距离计算distance，根据lr_pairs.csv得到mechanism。细胞数量大时无法储存为csv，可以考虑存为Parquet/npz

"""
========================================================
cell type level metrics
========================================================
目标（Goal）
-----------

本脚本用于在 demo 数据上复现论文中 “commonly identified interactions（共同相互作用）” 的评估流程：

1) 读取多个工具的标准化预测边（Pred）以及共享边（Shared）。
2) 对每个 (dataset, sample, tool) 计算相对于 Shared 的 TP / FP / FN。
3) 进一步计算每个 (dataset, sample, tool) 的 Precision / Recall / F1。
4) 使用箱线图（boxplot）在样本维度上展示各工具的 Precision / Recall / F1 分布（按 dataset 分开画）。

输入（Inputs）
--------------

1. pred_edges.csv（各工具预测结果；long-format 长表）
   必需字段（columns）：

     - dataset_id: str
     - sample_id:  str
     - tool:       str
     - sender_ct:  str
     - receiver_ct:str
     - ligand:     str
     - receptor:   str

   概念上的格式 / 维度：
     对每一个 (dataset_id, sample_id, tool)，该文件定义了一组“标准化边”的集合（SET）：
       Pred[(dataset, sample, tool)] = {
           (sender_ct, receiver_ct, ligand, receptor),
           ...
       }
     即：“细胞类型对 × 配体-受体（L–R）” 的边（用集合形式表达）。

2. shared_edges.csv（替代正例集 / 共享边；long-format 长表）
   必需字段（columns）：

     - dataset_id: str
     - sample_id:  str
     - sender_ct:  str
     - receiver_ct:str
     - ligand:     str
     - receptor:   str

   概念上的格式 / 维度：
     对每一个 (dataset_id, sample_id)，该文件定义了一组“标准化边”的集合（SET）：
       Shared[(dataset, sample)] = {
           (sender_ct, receiver_ct, ligand, receptor),
           ...
       }

   说明（Note）：
     在论文原始定义中，Shared 通常是由多个工具的输出统计得到的
     （例如：在 >=3 个工具中出现的边）。本 demo 为了让评估流程更清晰、可复现，
     直接把 Shared 作为输入文件提供。

输出（Outputs）
--------------

1. metrics（pandas DataFrame），每一行对应一个 (dataset_id, sample_id, tool)：
   输出字段包括：

     - TP, FP, FN: int
     - precision, recall, f1: float
     - n_pred: int    （该工具在该样本上预测的边数）
     - n_shared: int  （该样本 Shared 中的边数）

   指标定义（对每个 dataset/sample/tool）：

     - TP = |Pred ∩ Shared|
     - FP = |Pred \\ Shared|
     - FN = |Shared \\ Pred|
     - Precision = TP / (TP + FP)
     - Recall    = TP / (TP + FN)
     - F1        = 2 * Precision * Recall / (Precision + Recall)

   输出维度（Dimensionality）：
     metrics 的行数约为：#datasets × #samples_per_dataset × #tools。

2. demo_cci_metrics.csv
   将 metrics 表格保存到本地的 CSV 文件。

3. 箱线图（Boxplots）
   对每个 dataset，会分别生成 3 张图（每张图单独绘制，不使用 subplot）：

     - precision across samples（按工具分组）
     - recall across samples（按工具分组）
     - f1 across samples（按工具分组）
       """



# %%
"""
========================================================
Single cell level metrics
========================================================

目标（Goal）
-----------
在缺少真实 cell–cell 通讯真值的情况下，用“空间机制先验”做弱监督评估：
- 接触型（contact）应更短程（距离更小）
- 分泌型（secretion）可更长程（距离更大）
比较不同 CCC 方法的输出是否符合上述距离尺度规律。

输入（Input）
------------
读取一个 CSV：edges_long.csv
路径（本 demo）：
  /Users/taochenyang/Downloads/demo_cci_data/demo2_single_cell/edges_long.csv

数据格式（长表；每行=一条 cell–cell 边）：
必需列：
  dataset_id, sample_id, method, sender_id, receiver_id,
  score, distance, mechanism(contact/secretion),
  sender_type, receiver_type
维度：
  (E, 10)，demo 里 E≈3600（1 dataset × 3 sample × 3 method × 20×20）

内部处理（Normalization / Adjacency）
-------------------------------------
1) 计算 score_norm：
   在每个 (dataset_id, sample_id, method, mechanism) 组内对 score 做 rank 归一化：
     score_norm = rank(score) / n
   用于消除不同方法 score 量纲差异。

2) 定义 is_adjacent（邻接边）：
   每个 sample 内用 distance 的 ADJ_QUANTILE 分位数作阈值：
     is_adjacent = 1 if distance <= quantile(distance, ADJ_QUANTILE)

输出（Metrics 表格）
-------------------
所有输出都保存在 OUTDIR：
  /Users/taochenyang/Downloads/demo_cci_data/demo2_single_cell/

(1) metric1_distance_separation.csv  (≈ #dataset×#sample×#method 行)
  - delta_median = median(dist_secretion) - median(dist_contact)
  - ks_stat / ks_pvalue：contact vs secretion 距离分布的 KS 检验
  - auc_contact_vs_secretion：用 (-distance) 区分 contact(1)/secretion(0) 的 AUROC

(2) metric2_decay_length.csv
  - lambda_contact / lambda_secretion：拟合指数衰减曲线得到的长度尺度
  - r2_contact / r2_secretion：拟合优度
  - lambda_ratio = lambda_secretion / lambda_contact

(3) metric2_decay_curve.parquet
  - decay 曲线用的分箱点：distance_bin_center vs mean_score_norm（含 n_edges）

(4) metric3_contact_adjacency_enrichment.csv
  - 仅对 contact：高分边(top HIGH_SCORE_Q) 在邻接边(is_adjacent=1)中的富集
  - odds_ratio + 95%CI, fisher_pvalue（以及 2×2 计数）

输出（Figures 图像说明）
-----------------------
Metric 1：
- fig_metric1_delta_median.png：各方法 delta_median 箱线图（跨 sample）
- fig_metric1_auc.png：各方法 AUC 箱线图（跨 sample）
- fig_metric1_ecdf_contact.png：固定 contact，每个方法一条 distance 的 ECDF 曲线
- fig_metric1_ecdf_secretion.png：固定 secretion，每个方法一条 distance 的 ECDF 曲线

Metric 2：
- fig_metric2_lambda_ratio.png：各方法 lambda_ratio 箱线图（跨 sample）
- fig_metric2_decay_all_methods_contact.png：固定 contact，所有方法的 decay 曲线同图比较
- fig_metric2_decay_all_methods_secretion.png：固定 secretion，所有方法的 decay 曲线同图比较

Metric 3：
- fig_metric3_forest_or.png：contact 邻接富集 OR（log 轴）+ 95%CI 的 forest 图
"""
# %%
