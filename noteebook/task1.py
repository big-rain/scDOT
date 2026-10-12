import os
import copy
import pickle
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from dataloader.dataset_MET import build_patchseq_data
import shap
from Models.no_paired_patchseq import UnpairedClusterOTAE3Mod
# 依赖 mne 绘制弦图 (pip install mne)
from mne_connectivity.viz import plot_connectivity_circle
from collections import defaultdict
import re
import pandas as pd
import anndata as ad
import scanpy as sc


import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from mne_connectivity.viz import plot_connectivity_circle

import matplotlib.pyplot as plt

# 全局配置为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"  # 保证导出 SVG 时文字在 AI 中可编辑

def color_cm(cmap_name, n_colors):
    colormap = cm.get_cmap(cmap_name, max(n_colors, 1))
    return [colormap(i) for i in range(max(n_colors, 1))]


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]



# class ExplainerWrapper_Clus(nn.Module):
#     """阶段一：多模态输入 -> 多模态联合聚类 Logits (专供 SHAP 解释与亚型预测)
#
#     由于各模态在各自隐空间独立对齐聚类中心，
#     联合 Logits 即为各模态聚类 Logits 的融合 (取均值/求和)。
#     """
#
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#
#     def forward(self, *x_list):
#         # 兼容 SHAP 传入的列表/元组格式
#         if len(x_list) == 1 and isinstance(x_list[0], (list, tuple)):
#             x_list = x_list[0]
#         x_rna, x_ephys, x_morph = x_list[0], x_list[1], x_list[2]
#
#         # 1. 各模态分别编码
#         z_rna = self.model.encoder_rna(x_rna)
#         z_ephys = self.model.encoder_ephys(x_ephys)
#         z_morph = self.model.encoder_morph(x_morph)
#
#         # 2. 计算各模态的聚类 Logits (负欧氏距离平方，越近值越大)
#         logits_rna, _ = self.model.compute_cluster_assignment(
#             z_rna, self.model.cluster_layer_rna
#         )
#         logits_ephys, _ = self.model.compute_cluster_assignment(
#             z_ephys, self.model.cluster_layer_ephys
#         )
#         logits_morph, _ = self.model.compute_cluster_assignment(
#             z_morph, self.model.cluster_layer_morph
#         )
#
#         # 3. 三模态联合 Logits (求平均)，输出形状: [Batch, num_clusters]
#         joint_logits = (logits_rna + logits_ephys + logits_morph) / 3.0
#         return joint_logits


class ExplainerWrapper_Trans(nn.Module):
    """阶段二：单源模态 (如 RNA) -> 跨模态生成目标特征 (如 Ephys / Morph)

    直接走模型的隐空间跨模态解码器 (Cross-Modal Translation 路径)。
    """

    def __init__(self, model, src_name="rna", tgt_name="ephys"):
        super().__init__()
        self.model = model
        self.src_name = src_name
        self.tgt_name = tgt_name

        # 绑定源模态的 Encoder
        if src_name == "rna":
            self.encoder = model.encoder_rna
        elif src_name == "ephys":
            self.encoder = model.encoder_ephys
        elif src_name == "morph":
            self.encoder = model.encoder_morph
        else:
            raise ValueError(f"未知的源模态: {src_name}")

        # 绑定目标模态的 Decoder
        if tgt_name == "rna":
            self.decoder = model.decoder_rna
        elif tgt_name == "ephys":
            self.decoder = model.decoder_ephys
        elif tgt_name == "morph":
            self.decoder = model.decoder_morph
        else:
            raise ValueError(f"未知的目标模态: {tgt_name}")

    def forward(self, x_src):
        # 1. 源模态编码进入隐空间
        z_src = self.encoder(x_src)

        # 2. 目标模态跨模态生成重构特征，输出形状: [Batch, tgt_feature_dim]
        rec_tgt = self.decoder(z_src)
        return rec_tgt


def convert_to_serializable(obj):
    """递归将 numpy 数组、numpy 数值等转换为原生 Python 类型以便 JSON 序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def export_task1_selected_features(all_type_features, shap_values_dict=None,
                                   modality_names=['RNA', 'Ephys', 'Morphology'],
                                   save_dir='./analysis_results'):
    """
    将 Task 1 弦图挑选出的核心特征及其元数据导出为 CSV 和 JSON
    """
    os.makedirs(save_dir, exist_ok=True)
    records = []

    for group_name, mod_dict in all_type_features.items():
        for mod_idx, feats in mod_dict.items():
            mod_idx_int = int(mod_idx)
            mod_name = modality_names[mod_idx_int] if mod_idx_int < len(modality_names) else f"Modality_{mod_idx_int}"

            # 确保 feats 转换为原生列表
            feat_list = feats.tolist() if isinstance(feats, np.ndarray) else list(feats)

            for rank, feat_name in enumerate(feat_list, start=1):
                rec = {
                    'Group': str(group_name),
                    'Modality': str(mod_name),
                    'Modality_Idx': mod_idx_int,
                    'Feature_Name': str(feat_name),
                    'Importance_Rank': rank
                }
                if shap_values_dict and (group_name, mod_name, feat_name) in shap_values_dict:
                    rec['SHAP_Value'] = float(shap_values_dict[(group_name, mod_name, feat_name)])

                records.append(rec)

    df_features = pd.DataFrame(records)

    # 1. 保存 CSV
    csv_path = os.path.join(save_dir, 'Task1_Selected_Features_Summary.csv')
    df_features.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 2. 转换成可序列化的字典后保存 JSON
    serializable_dict = convert_to_serializable(all_type_features)
    json_path = os.path.join(save_dir, 'Task1_Selected_Features.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ 核心特征信息已成功保存至:\n  - CSV 表格: {csv_path}\n  - JSON 文件: {json_path}")
    return df_features

def export_chord_regulation_pairs(
    cv_all,
    io_names_all,
    src_name,
    tgt_name,
    cv_idx,
    save_dir="./analysis_results",
):
  """将跨模态弦图的所有连线特征及其 SHAP 调控强度导出为完整 CSV。"""
  os.makedirs(save_dir, exist_ok=True)
  records = []

  for ct, mat in cv_all.items():
    if ct not in io_names_all:
      continue
    in_names, out_names = io_names_all[ct]

    # 遍历矩阵中的每个 RNA ➔ Target 连线
    for i, src_feat in enumerate(in_names):
      for j, tgt_feat in enumerate(out_names):
        weight = float(mat[i, j])
        records.append({
            "Group": str(ct),
            "Source_Modality": src_name.upper(),
            "Source_Feature": str(src_feat),
            "Target_Modality": tgt_name.upper(),
            "Target_Feature": str(tgt_feat),
            "Regulation_Strength": weight,
            "Pair_Name": f"{src_feat} -> {tgt_feat}",
        })

  df_pairs = pd.DataFrame(records)

  # 按照每个 Group 内部的调控强度降序排序
  if not df_pairs.empty:
    df_pairs = df_pairs.sort_values(
        by=["Group", "Regulation_Strength"], ascending=[True, False]
    ).reset_index(drop=True)

  # 保存所有连线配对
  csv_path = os.path.join(
      save_dir, f"Task1_Regulation_All_Pairs_{src_name}2{tgt_name}_cv{cv_idx}.csv"
  )
  df_pairs.to_csv(csv_path, index=False, encoding="utf-8-sig")

  print(
      f"  💾 所有跨模态调控配对已保存至: {csv_path} (共 {len(df_pairs)} 条)"
  )

  return df_pairs


# def export_chord_regulation_pairs(
#     cv_all, io_names_all, src_name, tgt_name, cv_idx, save_dir
# ):
#   """将弦图返回的矩阵转换为结构化 CSV 表格 (Source_Gene -> Target_Ephys/Morph -> Weight)"""
#   os.makedirs(save_dir, exist_ok=True)
#   rows = []
#
#   if isinstance(cv_all, dict):
#     for subtype, mat in cv_all.items():
#       src_feats = (
#           io_names_all[subtype]["in"]
#           if isinstance(io_names_all, dict)
#           else io_names_all[0]
#       )
#       tgt_feats = (
#           io_names_all[subtype]["out"]
#           if isinstance(io_names_all, dict)
#           else io_names_all[1]
#       )
#       mat_arr = np.array(mat)
#       for r, s_feat in enumerate(src_feats):
#         for c, t_feat in enumerate(tgt_feats):
#           w = mat_arr[r, c] if r < mat_arr.shape[0] and c < mat_arr.shape[1] else 0.0
#           if abs(w) > 1e-6:
#             rows.append({
#                 "Subtype": subtype,
#                 "Source_Modality": src_name,
#                 "Source_Feature": s_feat,
#                 "Target_Modality": tgt_name,
#                 "Target_Feature": t_feat,
#                 "Regulation_Weight": w,
#             })
#   elif isinstance(cv_all, (np.ndarray, list)):
#     mat_arr = np.array(cv_all)
#     src_feats, tgt_feats = io_names_all[0], io_names_all[1]
#     for r, s_feat in enumerate(src_feats):
#       for c, t_feat in enumerate(tgt_feats):
#         w = mat_arr[r, c] if r < mat_arr.shape[0] and c < mat_arr.shape[1] else 0.0
#         if abs(w) > 1e-6:
#           rows.append({
#               "Source_Modality": src_name,
#               "Source_Feature": s_feat,
#               "Target_Modality": tgt_name,
#               "Target_Feature": t_feat,
#               "Regulation_Weight": w,
#           })
#
#   out_df = pd.DataFrame(rows)
#   if not out_df.empty:
#     out_df = out_df.sort_values(by="Regulation_Weight", ascending=False)
#   csv_file = os.path.join(
#       save_dir, f"Regulation_Pairs_{src_name}2{tgt_name}_cv{cv_idx}.csv"
#   )
#   out_df.to_csv(csv_file, index=False)
#   print(f"  📄 跨模态调控关系表格已保存: {csv_file} (共 {len(out_df)} 条)")


def align_clusters_to_metadata(pred_clusters, fine_encoded, fine_classes):
    """
    参考气泡图对齐标准：
    将 15 个预测 Cluster ID (0~14) 对齐并重命名为大类下细分的生物学名称 (如 Sst-1-Uni, Pvalb-2-Uni 等)
    """
    # 1. 构建 DataFrame
    df = pd.DataFrame({
        'pred_raw': [int(x) for x in pred_clusters],
        'true_name': [fine_classes[y] for y in fine_encoded]
    })

    # 2. 交叉统计表 (预测簇 x 真实细分类别)
    ct = pd.crosstab(df['pred_raw'], df['true_name'])

    # 按自然排序对齐细分类列
    ordered_fine = sorted([c for c in fine_classes if c in ct.columns], key=natural_sort_key)
    ct = ct.reindex(columns=ordered_fine, fill_value=0)

    col_idx_map = {col: idx for idx, col in enumerate(ct.columns)}
    col_weights = np.arange(ct.shape[1])

    # 3. 按主要组成类别及加权质心排序
    def get_sort_key(row_id):
        row_vals = ct.loc[row_id].values
        primary_col = ct.loc[row_id].idxmax()
        weighted_center = np.average(col_weights, weights=row_vals) if row_vals.sum() > 0 else 0
        return (col_idx_map[primary_col], weighted_center)

    sorted_raw_clusters = sorted(ct.index, key=get_sort_key)

    # 4. 统计大类并依序分配规范生物学命名
    coarse_counters = defaultdict(int)
    clus_to_name = {}  # key 是整数 cluster id (0 ~ 14)

    for raw_id in sorted_raw_clusters:
        # 该聚类中最多的真实细分类
        primary_fine = ct.loc[raw_id].idxmax()
        coarse = primary_fine.split('-')[0]  # 如 Pvalb, Sst
        coarse_counters[coarse] += 1
        clus_to_name[raw_id] = f"{coarse}-{coarse_counters[coarse]}-Uni"

    # 5. 生成各大类字典及样本级对齐标签
    major_dict = {name: name.split('-')[0] for name in clus_to_name.values()}
    aligned_sample_names = np.array([clus_to_name[k] for k in pred_clusters])

    return clus_to_name, major_dict, aligned_sample_names



import numpy as np
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    import torch
except ImportError:
    torch = None


def extract_subtype_markers_optimized(
        all_tensors,
        feature_names_list,
        pred_clusters,
        type_shap_list,
        val_indices,
        modality_names=("rna", "ephys", "morph"),
        top_n=7,
        min_cluster_n=4,  # 调整为更切合小样本的默认值 4
        min_out_n=5,
        min_pct_in=0.20,
        min_pct_diff=0.10,
        rna_min_log2fc=0.50,
        cont_min_effect=0.30,
        fdr_cutoff=0.05,
        min_shap_pos_frac=0.50,
        include_down_markers=True,
        fallback_top_n=None,
        shap_input="auto",
):
    """
    针对 Patch-seq 多模态亚型筛选特异性 Marker 的稳健实现。

    兼容下游弦图接口: results[cluster_key][modality]["indices"]
    """
    pred_clusters = np.asarray(pred_clusters)
    val_indices = np.asarray(val_indices, dtype=int)

    if pred_clusters.ndim != 1:
        raise ValueError("pred_clusters 必须是一维数组")

    if len(val_indices) == 0:
        raise ValueError("val_indices 不能为空")

    if len(all_tensors) != len(feature_names_list) or len(all_tensors) != len(type_shap_list):
        raise ValueError("all_tensors、feature_names_list 与 type_shap_list 长度必须一致")

    sub_clusters = pred_clusters[val_indices]
    unique_clusters = np.sort(np.unique(sub_clusters))
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError("至少需要两个聚类亚型")

    # 自动探测验证集最小簇大小，防止全空
    min_observed_n = min(np.sum(sub_clusters == c) for c in unique_clusters)
    if min_observed_n < min_cluster_n:
        print(f"[提示] 验证集中最小簇样本数仅为 {min_observed_n}，自动将 min_cluster_n 调整为 {min_observed_n}")
        min_cluster_n = min_observed_n

    results = {str(cluster): {} for cluster in unique_clusters}

    def to_numpy(x):
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def pooled_cohens_d(x_in, x_out):
        n_in = x_in.shape[0]
        n_out = x_out.shape[0]
        var_in = np.var(x_in, axis=0, ddof=1) if n_in > 1 else np.zeros(x_in.shape[1])
        var_out = np.var(x_out, axis=0, ddof=1) if n_out > 1 else np.zeros(x_out.shape[1])

        pooled_var = ((n_in - 1) * var_in + (n_out - 1) * var_out) / max(n_in + n_out - 2, 1)
        pooled_sd = np.sqrt(np.maximum(pooled_var, 0.0))
        mean_diff = np.mean(x_in, axis=0) - np.mean(x_out, axis=0)

        effect = np.zeros_like(mean_diff, dtype=float)
        valid = pooled_sd > 1e-8
        effect[valid] = mean_diff[valid] / pooled_sd[valid]

        zero_sd = ~valid
        effect[zero_sd] = np.sign(mean_diff[zero_sd]) * 1e6
        return effect

    def make_marker_dict(
            indices,
            feature_names,
            effect,
            p_vals,
            q_vals,
            mean_shap,
            shap_pos_frac,
            pct_in=None,
            pct_out=None,
            select_n=7,
    ):
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return []

        # 评分机制：兼顾效应量绝对值、平均 SHAP 绝对贡献与正向比例
        score = (
                np.abs(effect[indices])
                * np.abs(mean_shap[indices])
                * (shap_pos_frac[indices] + 0.1)
        )
        order = indices[np.argsort(score)[::-1][:select_n]]

        marker_list = []
        for i in order:
            item = {
                "name": str(feature_names[i]),
                "index": int(i),
                "effect": float(effect[i]),
                "p_value": float(p_vals[i]),
                "q_value": float(q_vals[i]),
                "mean_shap": float(mean_shap[i]),
                "shap_pos_frac": float(shap_pos_frac[i]),
            }
            if pct_in is not None:
                item["pct_in"] = float(pct_in[i])
                item["pct_out"] = float(pct_out[i])
            marker_list.append(item)
        return marker_list

    for mod_idx, (tensor, feature_names, shap_arr) in enumerate(
            zip(all_tensors, feature_names_list, type_shap_list)
    ):
        modality = modality_names[mod_idx] if mod_idx < len(modality_names) else f"mod_{mod_idx}"
        is_rna = modality.lower() == "rna"

        X_all = to_numpy(tensor)
        if X_all.ndim != 2:
            raise ValueError(f"{modality}: 输入必须是二维矩阵 [n_samples, n_features]")
        if X_all.shape[0] != len(pred_clusters):
            raise ValueError(f"{modality}: tensor 样本数与 pred_clusters 长度不一致")

        feature_names = np.asarray(feature_names).astype(str)
        if X_all.shape[1] != len(feature_names):
            raise ValueError(f"{modality}: 特征列数与 feature_names 数量不一致")

        X = X_all[val_indices].astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        shap = to_numpy(shap_arr)

        if shap.ndim != 3:
            raise ValueError(f"{modality}: SHAP 必须为 3D 矩阵 [samples, features, clusters]")

        n_features = X.shape[1]

        # SHAP 样本对齐
        if shap_input == "full":
            shap = shap[val_indices]
        elif shap_input == "val":
            if shap.shape[0] != len(val_indices):
                raise ValueError(f"{modality}: shap_input='val' 但样本数与 val_indices 不一致")
        elif shap_input == "auto":
            if shap.shape[0] == len(pred_clusters):
                shap = shap[val_indices]
            elif shap.shape[0] != len(val_indices):
                raise ValueError(f"{modality}: 无法自动对齐 SHAP 样本维度")

        # SHAP 维度转置适配: 期望为 [N, Features, Clusters]
        if shap.shape[1] != n_features:
            if shap.shape[2] == n_features and shap.shape[1] == n_clusters:
                shap = np.transpose(shap, (0, 2, 1))
            else:
                raise ValueError(f"{modality}: SHAP 特征维度与输入不匹配")

        for c_idx, cluster in enumerate(unique_clusters):
            cluster_key = str(cluster)
            in_mask = (sub_clusters == cluster)
            out_mask = ~in_mask

            n_in = int(np.sum(in_mask))
            n_out = int(np.sum(out_mask))

            if n_in < min_cluster_n or n_out < min_out_n:
                results[cluster_key][modality] = {
                    "names": [],
                    "indices": np.array([], dtype=int),
                    "up_markers": [],
                    "down_markers": [],
                    "candidates": [],
                    "is_fallback": False,
                    "n_in": n_in,
                    "n_out": n_out,
                    "reason": f"样本数不足: n_in={n_in}, n_out={n_out}",
                }
                continue

            X_in = X[in_mask]
            X_out = X[out_mask]
            shap_in = shap[in_mask, :, c_idx]

            mean_in = np.mean(X_in, axis=0)
            mean_out = np.mean(X_out, axis=0)
            mean_shap = np.mean(shap_in, axis=0)
            shap_pos_frac = np.mean(shap_in > 0, axis=0)

            if is_rna:
                pct_in = np.mean(X_in > 0, axis=0)
                pct_out = np.mean(X_out > 0, axis=0)
                pct_diff = pct_in - pct_out
                effect = np.log2((mean_in + 1.0) / (mean_out + 1.0))
            else:
                pct_in, pct_out, pct_diff = None, None, None
                effect = pooled_cohens_d(X_in, X_out)

            # 统计检验（统一使用 two-sided，保证上调和下调 Marker 均能计算出正确显著性）
            p_vals = np.ones(n_features, dtype=float)
            for fi in range(n_features):
                # 两组完全相等的常量特征跳过计算
                if np.all(X_in[:, fi] == X_out[0, fi]) and np.all(X_out[:, fi] == X_out[0, fi]):
                    continue
                try:
                    _, p = stats.mannwhitneyu(
                        X_in[:, fi],
                        X_out[:, fi],
                        alternative="two-sided",
                    )
                    if np.isfinite(p):
                        p_vals[fi] = p
                except Exception:
                    p_vals[fi] = 1.0

            # 多重假设检验校正
            q_vals = multipletests(p_vals, method="fdr_bh")[1]

            # 筛选条件
            sig_mask = q_vals <= fdr_cutoff
            shap_mask = (mean_shap > 0) & (shap_pos_frac >= min_shap_pos_frac)

            if is_rna:
                up_mask = (
                        sig_mask
                        & shap_mask
                        & (effect >= rna_min_log2fc)
                        & (pct_in >= min_pct_in)
                        & (pct_diff >= min_pct_diff)
                )
                down_mask = (
                        sig_mask
                        & (effect <= -rna_min_log2fc)
                )
            else:
                up_mask = sig_mask & shap_mask & (effect >= cont_min_effect)
                down_mask = sig_mask & (effect <= -cont_min_effect)

            if not include_down_markers:
                down_mask[:] = False

            up_markers = make_marker_dict(
                np.where(up_mask)[0],
                feature_names, effect, p_vals, q_vals, mean_shap, shap_pos_frac,
                pct_in=pct_in, pct_out=pct_out, select_n=top_n
            )
            down_markers = make_marker_dict(
                np.where(down_mask)[0],
                feature_names, effect, p_vals, q_vals, mean_shap, shap_pos_frac,
                pct_in=pct_in, pct_out=pct_out, select_n=top_n
            )

            strict_indices = [m["index"] for m in up_markers] + [m["index"] for m in down_markers]
            strict_indices = np.asarray(sorted(set(strict_indices)), dtype=int)
            strict_names = [str(feature_names[i]) for i in strict_indices]

            # Fallback 策略：如果严格阈值下无 Marker，按效应量与 SHAP 综合得分选出候选特征，确保弦图不崩
            candidates = []
            is_fallback = False

            if len(strict_indices) == 0:
                is_fallback = True
                fallback_n = fallback_top_n or top_n
                candidate_score = np.abs(effect) * np.abs(mean_shap) * (shap_pos_frac + 0.1)

                # 至少要求有表达或具有非零方差
                valid_cand = np.ones(n_features, dtype=bool)
                if is_rna:
                    valid_cand = (pct_in >= (min_pct_in * 0.5))

                candidate_indices = np.where(valid_cand)[0]
                if len(candidate_indices) > 0:
                    sorted_cand = candidate_indices[np.argsort(candidate_score[candidate_indices])[::-1][:fallback_n]]
                    candidates = make_marker_dict(
                        sorted_cand,
                        feature_names, effect, p_vals, q_vals, mean_shap, shap_pos_frac,
                        pct_in=pct_in, pct_out=pct_out, select_n=fallback_n
                    )
                    strict_indices = np.asarray([m["index"] for m in candidates], dtype=int)
                    strict_names = [str(feature_names[i]) for i in strict_indices]

            results[cluster_key][modality] = {
                "names": strict_names,
                "indices": strict_indices,
                "up_markers": up_markers,
                "down_markers": down_markers,
                "candidates": candidates,
                "is_fallback": is_fallback,
                "n_in": n_in,
                "n_out": n_out,
                "all_stats": {
                    "feature_names": feature_names.tolist(),
                    "effect": effect,
                    "p_values": p_vals,
                    "q_values": q_vals,
                    "mean_shap": mean_shap,
                    "shap_pos_frac": shap_pos_frac,
                    "pct_in": pct_in,
                    "pct_out": pct_out,
                    "strict_up_mask": up_mask,
                    "strict_down_mask": down_mask,
                },
            }

        print(f"✓ [{modality.upper()}] marker 提取完成")

    return results
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 确保全局字体为 Times New Roman 且 SVG 原生文字可编辑
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"


def plot_type_chord_by_major(
    shap_values,
    feature_names_list,
    p_fe_idx,
    clus_to_name,
    major_dict,
    save_dir,
    thres=7,
    font_size=20,  # ✅ 新增参数：默认大字号 15pt
):
    """阶段一绘图：按大类 (Pvalb, Sst 等) 导出各自细分亚型与生物特征的弦图"""
    os.makedirs(save_dir, exist_ok=True)
    scaler = MinMaxScaler()
    potential_colormaps = ["spring", "summer", "winter"]
    num_modalities = len(feature_names_list)

    colors_mods = [
        color_cm(potential_colormaps[ii % 3], len(feat_list))
        for ii, feat_list in enumerate(feature_names_list)
    ]
    all_major_types = np.unique(list(major_dict.values()))

    cmap_pts = plt.get_cmap("tab20")
    unique_subtypes = list(clus_to_name.values())
    colors_type = {
        name: cmap_pts(i % 20) for i, name in enumerate(unique_subtypes)
    }

    all_type_features = {}
    max_available_c = shap_values[0].shape[2]

    for maj_t in all_major_types:
        sub_clusters = [
            c_id
            for c_id, name in clus_to_name.items()
            if major_dict[name] == maj_t and c_id < max_available_c
        ]
        if len(sub_clusters) == 0:
            continue

        edges = []

        for u_idx, tp in enumerate(sub_clusters):
            tp_name = clus_to_name[tp]
            all_type_features[tp_name] = {}

            for m_idx, feat_names in enumerate(feature_names_list):
                raw_mat = shap_values[m_idx][:, :, tp]
                mean_shap = np.mean(np.abs(raw_mat), axis=0)

                # 归一化重要性
                valid_idx = np.array(p_fe_idx[m_idx])
                ft = scaler.fit_transform(
                    mean_shap[valid_idx].reshape(-1, 1)
                ).squeeze()

                # 选取 Top-K 重要特征
                k_val = min(thres, len(ft))
                top_k_sub_indices = np.argsort(ft)[-k_val:]

                selected_feat_names = []
                for sub_i in top_k_sub_indices:
                    w = ft[sub_i]
                    orig_feat_idx = valid_idx[sub_i]
                    f_name = feat_names[orig_feat_idx]

                    selected_feat_names.append(f_name)
                    edges.append(
                        {
                            "feat": f_name,
                            "subtype": tp_name,
                            "weight": w,
                            "mod_idx": m_idx,
                            "orig_idx": orig_feat_idx,
                        }
                    )

                all_type_features[tp_name][m_idx] = np.array(
                    selected_feat_names[::-1]
                )

        if len(edges) == 0:
            continue

        active_subtypes = [clus_to_name[tp] for tp in sub_clusters]

        active_feats = []
        feat_to_mod = {}
        feat_to_orig_idx = {}
        for e in edges:
            f = e["feat"]
            if f not in active_feats:
                active_feats.append(f)
                feat_to_mod[f] = e["mod_idx"]
                feat_to_orig_idx[f] = e["orig_idx"]

        active_feats = sorted(
            active_feats, key=lambda f: (feat_to_mod[f], feat_to_orig_idx[f])
        )

        all_nodes = active_feats + active_subtypes
        node_to_idx = {name: idx for idx, name in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        n_feats = len(active_feats)
        n_subs = len(active_subtypes)

        con_weights = np.array([e["weight"] for e in edges])
        src_indices = np.array([node_to_idx[e["feat"]] for e in edges])
        dst_indices = np.array([node_to_idx[e["subtype"]] for e in edges])

        feat_angles = (
            np.linspace(45, 315, n_feats).astype(int) % 360
            if n_feats > 0
            else np.array([])
        )
        sub_angles = (
            np.linspace(-15, 15, n_subs).astype(int) % 360
            if n_subs > 1
            else np.array([0])
        )
        node_angles = np.concatenate([feat_angles, sub_angles])

        node_colors = []
        for f in active_feats:
            m = feat_to_mod[f]
            orig_i = feat_to_orig_idx[f]
            node_colors.append(colors_mods[m][orig_i])
        for st in active_subtypes:
            node_colors.append(colors_type[st])
        node_colors = np.array(node_colors)

        fig, _ = plot_connectivity_circle(
            con_weights,
            all_nodes,
            indices=(src_indices, dst_indices),
            node_width=5,
            node_linewidth=0,
            node_colors=node_colors,
            facecolor="white",
            textcolor="black",
            linewidth=1.2,
            node_angles=node_angles,
            colormap="Reds",
            vmin=0.2,
            vmax=1.0,
            show=False,
            fontsize_names=font_size,  # 增大字号
        )
        fig.set_size_inches(12, 12)  # 同步扩大画布，防止长名字截断

        safe_maj_t = maj_t.replace("/", "-")
        save_svg = os.path.join(save_dir, f"Type_Chord_{safe_maj_t}.svg")

        # ✅【关键优化 3】：pad_inches=0.3 留足四周外边距
        fig.savefig(
            save_svg,
            bbox_inches="tight",
            pad_inches=0.3,
            format="svg",
        )
        plt.close(fig)
        print(f"📊 [Type 弦图保存成功(大字号版)] -> {save_svg}")

    return all_type_features


def feature_relevance_chord_plot(
    shap_values_0_1,
    unique_ct,
    var_names_all,
    all_type_features,
    technique,
    in_mod=0,
    direction="0to1",
    thres=None,
    fontsize_names=20,  # ✅ 默认调大至 15pt
    potential_coloarmaps=["spring", "summer", "winter", "autumn"],
):
    os.makedirs("./analysis_results/feature_relevance_figures", exist_ok=True)
    scaler = MinMaxScaler()

    # 解析源和目标模态编号
    dir_parts = direction.split("to")
    direction_dict = {0: int(dir_parts[0]), 1: int(dir_parts[1])}

    # 为各模态生成基准颜色
    colors_mods = [
        color_cm(
            potential_coloarmaps[
                direction_dict[ii] % len(potential_coloarmaps)
            ],
            len(c_p_fe),
        )
        for ii, c_p_fe in enumerate(var_names_all)
    ]
    target_mod = 1 if in_mod == 0 else 0

    cv_all = {}
    io_names_all = {}

    # 1. 统一整理 SHAP 矩阵 -> (len(var_names_all[0]), len(var_names_all[1]))
    if isinstance(shap_values_0_1, np.ndarray) and shap_values_0_1.ndim == 3:
        shap_mean_matrix = np.mean(
            np.abs(shap_values_0_1), axis=0
        )  # (N_src, N_tgt)
    elif isinstance(shap_values_0_1, list):
        shap_mean_matrix = np.column_stack(
            [np.mean(np.abs(sp), axis=0) for sp in shap_values_0_1]
        )
    else:
        shap_mean_matrix = np.array(shap_values_0_1)

    # 确保行对应源模态，列对应目标模态
    if shap_mean_matrix.shape[0] != len(var_names_all[in_mod]):
        shap_mean_matrix = shap_mean_matrix.T

    # 归一化权重至 [0, 1]
    shap_mean_norm = scaler.fit_transform(shap_mean_matrix)

    # 2. 遍历各细胞亚型分别绘制
    for ct in unique_ct:
        if ct not in all_type_features:
            continue

        cur_marker, cur_idx = [], []
        for ii_m, v_n in enumerate(var_names_all):
            tgt_feats = all_type_features[ct].get(direction_dict[ii_m], [])
            _m, _i, _ = np.intersect1d(v_n, tgt_feats, return_indices=True)
            cur_marker.append(_m)
            cur_idx.append(_i)

        # 检查是否有足够的特征可供画图
        if len(cur_marker[0]) == 0 or len(cur_marker[1]) == 0:
            print(f"⚠️ [跳过] 亚型 {ct} 未找到足够的匹配特征节点")
            continue

        if thres is not None:
            n_in, n_out = np.where(shap_mean_norm > thres)
            if len(n_in) == 0:
                continue
            sub_norm = shap_mean_norm[np.unique(n_in), :]
            c_v_in, c_v_out = np.where(sub_norm > thres)
            c_v = sub_norm[sub_norm > thres]
            fe_shape = sub_norm.shape
            in_names = var_names_all[0][np.unique(n_in)]
            out_names = var_names_all[1][np.unique(n_out)]
            names_idx = [np.unique(n_in), np.unique(n_out)]
        else:
            # 提取属于该细胞类型的特定 Marker 基因与特征间的关联系数
            c_v = shap_mean_norm[cur_idx[in_mod], :][:, cur_idx[target_mod]]
            fe_shape = c_v.shape
            c_v_in, c_v_out = np.where(c_v >= 0)
            c_v = c_v.ravel()
            in_names, _in_idx, _ = np.intersect1d(
                var_names_all[0], cur_marker[0], return_indices=True
            )
            out_names, _out_idx, _ = np.intersect1d(
                var_names_all[1], cur_marker[1], return_indices=True
            )
            names_idx = [_in_idx, _out_idx]

        io_names = [in_names, out_names]
        node_names = list(in_names) + list(out_names)
        c_v_out = c_v_out + fe_shape[0]  # 目标模态节点索引偏移

        starts = [105, -75]
        ends = [255, 75]

        node_angles = []
        node_colors_list = []

        for ii, nm in enumerate(io_names):
            start = starts[ii]
            end = ends[ii]
            n_nodes_mod = len(nm)

            # 分配角度
            angles = (np.linspace(start, end, n_nodes_mod)).astype(int) % 360
            node_angles.append(angles)

            # 颜色是 RGBA (4维)，保持为独立的颜色元组列表
            mod_colors = [colors_mods[ii][idx] for idx in names_idx[ii]]
            node_colors_list.extend(mod_colors)

        node_angles = np.concatenate(node_angles)
        node_colors = node_colors_list

        # ✅【关键优化 1】：扩大字号为 fontsize_names
        fig, axes = plot_connectivity_circle(
            con=c_v,
            node_names=node_names,
            indices=(c_v_in, c_v_out),
            node_width=5,
            node_linewidth=0,
            node_colors=node_colors,
            facecolor="white",
            textcolor="k",
            linewidth=2,
            node_angles=node_angles,
            colormap="Reds",
            vmin=0.0,
            vmax=1.0,
            show=False,
            fontsize_names=fontsize_names,  # 大字号
        )

        # ✅【关键优化 2】：画布扩大到 12x12，给两侧文字留出充足空间
        fig.set_size_inches(12, 12)

        safe_ct = str(ct).replace("/", "-").replace(" ", "_")
        save_file = f"./analysis_results/feature_relevance_figures/{technique}_feature_{safe_ct}_only_show_good_{direction}.svg"

        # ✅【关键优化 3】：pad_inches=0.3 留出边缘边距，防止长文本裁切
        fig.savefig(
            save_file,
            bbox_inches="tight",
            pad_inches=0.3,
            format="svg",
        )
        plt.close(fig)

        cv_all[ct] = c_v.reshape(fe_shape)
        io_names_all[ct] = io_names
        print(f"📊 [跨模态关联弦图保存成功(大字版)] -> {save_file}")

    return cv_all, io_names_all


# def feature_relevance_chord_plot(shap_values_0_1, unique_ct, var_names_all, all_type_features, technique, in_mod=0,
#                                  direction='0to1', thres=None, fontsize_names=12,
#                                  potential_coloarmaps=['spring', 'summer', 'winter', 'autumn']):
#     os.makedirs('./feature_relevance_figures', exist_ok=True)
#     scaler = MinMaxScaler()
#
#     # 解析源和目标模态编号
#     dir_parts = direction.split('to')
#     direction_dict = {0: int(dir_parts[0]), 1: int(dir_parts[1])}
#
#     # 为各模态生成基准颜色
#     colors_mods = [
#         color_cm(potential_coloarmaps[direction_dict[ii] % len(potential_coloarmaps)], len(c_p_fe))
#         for ii, c_p_fe in enumerate(var_names_all)
#     ]
#     target_mod = 1 if in_mod == 0 else 0
#
#     cv_all = {}
#     io_names_all = {}
#
#     # 1. 统一整理 SHAP 矩阵 -> (len(var_names_all[0]), len(var_names_all[1]))
#     if isinstance(shap_values_0_1, np.ndarray) and shap_values_0_1.ndim == 3:
#         shap_mean_matrix = np.mean(np.abs(shap_values_0_1), axis=0)  # (N_src, N_tgt)
#     elif isinstance(shap_values_0_1, list):
#         shap_mean_matrix = np.column_stack([np.mean(np.abs(sp), axis=0) for sp in shap_values_0_1])
#     else:
#         shap_mean_matrix = np.array(shap_values_0_1)
#
#     # 确保行对应源模态，列对应目标模态
#     if shap_mean_matrix.shape[0] != len(var_names_all[in_mod]):
#         shap_mean_matrix = shap_mean_matrix.T
#
#     # 归一化权重至 [0, 1]
#     shap_mean_norm = scaler.fit_transform(shap_mean_matrix)
#
#     # 2. 遍历各细胞亚型分别绘制
#     for ct in unique_ct:
#         if ct not in all_type_features:
#             continue
#
#         cur_marker, cur_idx = [], []
#         for ii_m, v_n in enumerate(var_names_all):
#             tgt_feats = all_type_features[ct].get(direction_dict[ii_m], [])
#             _m, _i, _ = np.intersect1d(v_n, tgt_feats, return_indices=True)
#             cur_marker.append(_m)
#             cur_idx.append(_i)
#
#         # 检查是否有足够的特征可供画图
#         if len(cur_marker[0]) == 0 or len(cur_marker[1]) == 0:
#             print(f"⚠️ [跳过] 亚型 {ct} 未找到足够的匹配特征节点")
#             continue
#
#         if thres is not None:
#             n_in, n_out = np.where(shap_mean_norm > thres)
#             if len(n_in) == 0:
#                 continue
#             sub_norm = shap_mean_norm[np.unique(n_in), :]
#             c_v_in, c_v_out = np.where(sub_norm > thres)
#             c_v = sub_norm[sub_norm > thres]
#             fe_shape = sub_norm.shape
#             in_names = var_names_all[0][np.unique(n_in)]
#             out_names = var_names_all[1][np.unique(n_out)]
#             names_idx = [np.unique(n_in), np.unique(n_out)]
#         else:
#             # 提取属于该细胞类型的特定 Marker 基因与特征间的关联系数
#             c_v = shap_mean_norm[cur_idx[in_mod], :][:, cur_idx[target_mod]]
#             fe_shape = c_v.shape
#             c_v_in, c_v_out = np.where(c_v >= 0)
#             c_v = c_v.ravel()
#             in_names, _in_idx, _ = np.intersect1d(var_names_all[0], cur_marker[0], return_indices=True)
#             out_names, _out_idx, _ = np.intersect1d(var_names_all[1], cur_marker[1], return_indices=True)
#             names_idx = [_in_idx, _out_idx]
#
#         io_names = [in_names, out_names]
#         node_names = list(in_names) + list(out_names)
#         c_v_out = c_v_out + fe_shape[0]  # 目标模态节点索引偏移
#
#         starts = [105, -75]
#         ends = [255, 75]
#
#         node_angles = []
#         node_colors_list = []
#
#         for ii, nm in enumerate(io_names):
#             start = starts[ii]
#             end = ends[ii]
#             n_nodes_mod = len(nm)
#
#             # 分配角度
#             angles = (np.linspace(start, end, n_nodes_mod)).astype(int) % 360
#             node_angles.append(angles)
#
#             # 💡【核心修复】：颜色是 RGBA (4维)，必须保持为独立的颜色元组列表，不能直接横向 hstack
#             mod_colors = [colors_mods[ii][idx] for idx in names_idx[ii]]
#             node_colors_list.extend(mod_colors)
#
#         node_angles = np.concatenate(node_angles)
#         node_colors = node_colors_list  # 保持为 list of (R, G, B, A)
#
#         s_title = str(ct)
#         fig, axes = plot_connectivity_circle(
#             con=c_v,
#             node_names=node_names,
#             indices=(c_v_in, c_v_out),
#             node_width=5,
#             node_linewidth=0,
#             node_colors=node_colors,  # 正确的 4 维 RGBA 颜色列表
#             facecolor='white',
#             textcolor='k',
#             linewidth=2,
#             node_angles=node_angles,
#             colormap='Reds',
#             vmin=0.0,
#             vmax=1.0,
#             show=False,
#             fontsize_names=fontsize_names,
#             # title=s_title
#         )
#         fig.set_size_inches(10, 10)
#
#         safe_ct = str(ct).replace('/', '-').replace(' ', '_')
#         # 后缀改为 .svg
#         save_file = f'./feature_relevance_figures/{technique}_feature_{safe_ct}_only_show_good_{direction}.svg'
#
#         # 指定 format='svg'（SVG 是矢量图，dpi 参数在保存矢量路径时不影响清晰度，但可保留）
#         fig.savefig(save_file, bbox_inches='tight', format='svg')
#         plt.close(fig)
#
#         cv_all[ct] = c_v.reshape(fe_shape)
#         io_names_all[ct] = io_names
#
#     return cv_all, io_names_all

#
# def plot_type_chord_by_major(
#     shap_values,
#     feature_names_list,
#     p_fe_idx,
#     clus_to_name,
#     major_dict,
#     save_dir,
#     thres=7,
# ):
#     """阶段一绘图：按大类 (Pvalb, Sst 等) 导出各自细分亚型与生物特征的弦图"""
#     os.makedirs(save_dir, exist_ok=True)
#     scaler = MinMaxScaler()
#     potential_colormaps = ["spring", "summer", "winter"]
#     num_modalities = len(feature_names_list)
#
#     colors_mods = [
#         color_cm(potential_colormaps[ii % 3], len(feat_list))
#         for ii, feat_list in enumerate(feature_names_list)
#     ]
#     all_major_types = np.unique(list(major_dict.values()))
#
#     cmap_pts = plt.get_cmap("tab20")
#     unique_subtypes = list(clus_to_name.values())
#     colors_type = {name: cmap_pts(i % 20) for i, name in enumerate(unique_subtypes)}
#
#     all_type_features = {}
#     max_available_c = shap_values[0].shape[2]
#
#     for maj_t in all_major_types:
#         sub_clusters = [
#             c_id
#             for c_id, name in clus_to_name.items()
#             if major_dict[name] == maj_t and c_id < max_available_c
#         ]
#         if len(sub_clusters) == 0:
#             continue
#
#         edges = []
#
#         for u_idx, tp in enumerate(sub_clusters):
#             tp_name = clus_to_name[tp]
#             all_type_features[tp_name] = {}
#
#             for m_idx, feat_names in enumerate(feature_names_list):
#                 raw_mat = shap_values[m_idx][:, :, tp]
#                 mean_shap = np.mean(np.abs(raw_mat), axis=0)
#
#                 # 归一化重要性
#                 valid_idx = np.array(p_fe_idx[m_idx])
#                 ft = scaler.fit_transform(mean_shap[valid_idx].reshape(-1, 1)).squeeze()
#
#                 # 选取 Top-K 重要特征
#                 k_val = min(thres, len(ft))
#                 top_k_sub_indices = np.argsort(ft)[-k_val:]
#
#                 selected_feat_names = []
#                 for sub_i in top_k_sub_indices:
#                     w = ft[sub_i]
#
#                     # ✅【核心修复】：映射回真实的全量特征索引 orig_feat_idx
#                     orig_feat_idx = valid_idx[sub_i]
#                     f_name = feat_names[orig_feat_idx]  # 正确取名
#
#                     selected_feat_names.append(f_name)
#                     edges.append(
#                         {
#                             "feat": f_name,
#                             "subtype": tp_name,
#                             "weight": w,
#                             "mod_idx": m_idx,
#                             "orig_idx": orig_feat_idx,  # 正确传递颜色和位置索引
#                         }
#                     )
#
#                 all_type_features[tp_name][m_idx] = np.array(
#                     selected_feat_names[::-1]
#                 )  # 按重要性降序排列
#
#         if len(edges) == 0:
#             continue
#
#         # 收集本次弦图中出现的所有独立特征和亚型节点
#         active_subtypes = [clus_to_name[tp] for tp in sub_clusters]
#
#         # 特征节点按模态分类排列，保持视觉有序
#         active_feats = []
#         feat_to_mod = {}
#         feat_to_orig_idx = {}
#         for e in edges:
#             f = e["feat"]
#             if f not in active_feats:
#                 active_feats.append(f)
#                 feat_to_mod[f] = e["mod_idx"]
#                 feat_to_orig_idx[f] = e["orig_idx"]
#
#         # 排序特征：按模态 0, 1, 2 排序
#         active_feats = sorted(
#             active_feats, key=lambda f: (feat_to_mod[f], feat_to_orig_idx[f])
#         )
#
#         # 总节点列表：先是特征，后面是亚型
#         all_nodes = active_feats + active_subtypes
#         node_to_idx = {name: idx for idx, name in enumerate(all_nodes)}
#         n_nodes = len(all_nodes)
#         n_feats = len(active_feats)
#         n_subs = len(active_subtypes)
#
#         # 构建连线权重与索引对
#         con_weights = np.array([e["weight"] for e in edges])
#         src_indices = np.array([node_to_idx[e["feat"]] for e in edges])
#         dst_indices = np.array([node_to_idx[e["subtype"]] for e in edges])
#
#         # 分配圆环角度: 特征占 45° ~ 315°，亚型占 -15° ~ 15°
#         feat_angles = (
#             np.linspace(45, 315, n_feats).astype(int) % 360
#             if n_feats > 0
#             else np.array([])
#         )
#         sub_angles = (
#             np.linspace(-15, 15, n_subs).astype(int) % 360
#             if n_subs > 1
#             else np.array([0])
#         )
#         node_angles = np.concatenate([feat_angles, sub_angles])
#
#         # 分配节点颜色
#         node_colors = []
#         for f in active_feats:
#             m = feat_to_mod[f]
#             orig_i = feat_to_orig_idx[f]
#             node_colors.append(colors_mods[m][orig_i])
#         for st in active_subtypes:
#             node_colors.append(colors_type[st])
#         node_colors = np.array(node_colors)
#
#         # 绘制圆形弦图
#         fig, _ = plot_connectivity_circle(
#             con_weights,
#             all_nodes,
#             indices=(src_indices, dst_indices),
#             node_width=5,
#             node_linewidth=0,
#             node_colors=node_colors,
#             facecolor="white",
#             textcolor="black",
#             linewidth=1.2,
#             node_angles=node_angles,
#             colormap="Reds",
#             vmin=0.2,
#             vmax=1.0,
#             show=False,
#             fontsize_names=10,
#         )
#         fig.set_size_inches(10, 10)
#
#         safe_maj_t = maj_t.replace("/", "-")
#         save_pdf = os.path.join(save_dir, f"Type_Chord_{safe_maj_t}.svg")
#         fig.savefig(save_pdf, bbox_inches="tight", format='svg')
#         plt.close(fig)
#         print(f"📊 [Type 弦图保存成功] -> {save_pdf}")
#
#     return all_type_features



import os
import pickle
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

# =================================================================
# 1. 定义专为 16 个亚型定制的单模态聚类解释器
class ExplainerWrapper_Clus(nn.Module):
    def __init__(self, encoder, centers_64):
        super().__init__()
        self.encoder = encoder
        self.register_buffer(
            "centers", torch.tensor(centers_64, dtype=torch.float32)
        )

    def forward(self, x):
        z = self.encoder(x)
        dist_sq = torch.sum(
            (z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=-1
        ).clamp(min=1e-8)
        return -dist_sq




def run_patchseq_shap_pipeline(
    model,
    data,
    ind_dict,
    cv,
    save_dir="./shap_chord_results",
    top_n_markers=50,
    latent_csv_path=r"D:\PycharmProjects\C1\Models\checkpoints_paired\patchseq0.512\patchseq_best_latent_z.csv",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./feature_relevance_figures", exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # 1. 元数据准备
    modality_names = ["rna", "ephys", "morph"]
    all_tensors = [data["all"]["tensors"][m] for m in modality_names]
    meta = data["meta"]
    fine_encoded = data["all"]["labels"]["fine"].cpu().numpy()
    fine_classes = meta["fine_classes"]
    feature_names_list = [meta["feature_names"][m] for m in modality_names]

    # -----------------------------------------------------------------
    # 2. 构造背景数据 (Background Prototypes)
    # -----------------------------------------------------------------
    val_indices = ind_dict[cv]["train"]
    train_indices = ind_dict[cv]["val"]
    bg_tensors = []
    unique_fine_ids = np.unique(fine_encoded[train_indices])

    for t_mod in all_tensors:
        proto_means = []
        for f_id in unique_fine_ids:
            idx_cls = np.intersect1d(
                np.where(fine_encoded == f_id)[0], train_indices
            )
            if len(idx_cls) > 0:
                proto_means.append(t_mod[idx_cls].mean(dim=0))
            else:
                proto_means.append(t_mod[train_indices].mean(dim=0))
        bg_tensors.append(torch.stack(proto_means, dim=0))
    print(f"✓ 背景模型初始化完成: {len(bg_tensors[0])} 个亚型原型")

    #  3. 从 CSV 读取 192 维特征及 16 个簇的真实聚类结果 (kmeans_pred)
    # -----------------------------------------------------------------
    if not os.path.exists(latent_csv_path):
        raise FileNotFoundError(f"未找到聚类结果文件: {latent_csv_path}")

    print(f"📖 正在从 CSV 加载聚类结果与隐空间特征: {latent_csv_path}")
    df_latent = pd.read_csv(latent_csv_path)

    # 读取聚类预测标签 (16 个簇)
    pred_clusters = df_latent["kmeans_pred"].values
    unique_clusters = np.unique(pred_clusters)
    n_real_clusters = len(unique_clusters)
    print(
        f"🎯 成功识别实际聚类簇数: {n_real_clusters} 个簇 (对应类别:"
        f" {unique_clusters})"
    )

    # 过滤非特征列，提取 192 维隐空间特征矩阵
    meta_col_names = [
        "sample_id",
        "cell_id",
        "true_label",
        "kmeans_pred",
        "label",
        "pred",
    ]
    feat_cols = [c for c in df_latent.columns if c not in meta_col_names]
    z_fused = df_latent[feat_cols].values  # [N, 192]
    print(f"✓ 成功提取 192 维特征矩阵，形状: {z_fused.shape}")

    # 计算这 16 个簇在 192 维空间中的实际中心
    centers_192 = np.zeros((n_real_clusters, z_fused.shape[1]))
    for i, c_id in enumerate(unique_clusters):
        mask = pred_clusters == c_id
        centers_192[i] = np.mean(z_fused[mask], axis=0)

    # 将 192 维中心切分成各模态对应的 64 维中心
    latent_dim = (
        model.latent_dim
        if hasattr(model, "latent_dim")
        else (z_fused.shape[1] // 3)
    )
    centers_dict = {
        "rna": centers_192[:, 0:latent_dim],
        "ephys": centers_192[:, latent_dim: latent_dim * 2],
        "morph": centers_192[:, latent_dim * 2: latent_dim * 3],
    }

    # -----------------------------------------------------------------
    # 4. 亚型聚类与元数据对齐
    # -----------------------------------------------------------------
    clus_to_name, major_dict, _ = align_clusters_to_metadata(
        pred_clusters, fine_encoded, fine_classes
    )

    # -----------------------------------------------------------------
    # 5. 阶段一：计算针对实际 16 个亚型的全量 Type Relevance SHAP
    # -----------------------------------------------------------------
    print(f"\n🚀 [1/2] 正在计算针对实际 {n_real_clusters} 个亚型的 Type Relevance SHAP...")
    test_tensors_val = [t_mod[val_indices] for t_mod in all_tensors]

    mod_encoders = {
        "rna": model.encoder_rna,
        "ephys": model.encoder_ephys,
        "morph": model.encoder_morph,
    }

    type_shap = []
    for mod_idx, mod_name in enumerate(modality_names):
        print(f"  ⚡ 正在解释模态: {mod_name.upper()} (特征数: {all_tensors[mod_idx].shape[1]})...")

        explainer_net = ExplainerWrapper_Clus(
            encoder=mod_encoders[mod_name], centers_64=centers_dict[mod_name]
        ).to(device)
        explainer_net.eval()

        bg_mod = bg_tensors[mod_idx].to(device)
        test_mod = test_tensors_val[mod_idx].to(device)

        explainer = shap.GradientExplainer(explainer_net, bg_mod)
        shap_vals = explainer.shap_values(test_mod)

        if isinstance(shap_vals, list):
            shap_arr = np.stack(shap_vals, axis=-1)
        else:
            shap_arr = np.array(shap_vals)

        type_shap.append(shap_arr)
        print(f"    ✓ {mod_name} SHAP 矩阵形状: {shap_arr.shape}")
        # 输出形状严格为: [len(val_indices), n_features, n_real_clusters]

    # 保存计算好的 SHAP 矩阵
    shap_save_path = os.path.join(save_dir, f"type_shap_cv{cv}.pkl")
    with open(shap_save_path, "wb") as f:
        pickle.dump(type_shap, f)
    print(f"✓ SHAP 矩阵已持久化保存至: {shap_save_path}")


    # 2. 调用稳健型 Marker 提取器
    selected_features_dict = extract_subtype_markers_optimized(
        all_tensors=all_tensors,
        feature_names_list=feature_names_list,
        pred_clusters=pred_clusters,
        type_shap_list=type_shap,
        val_indices=val_indices,
        modality_names=["rna", "ephys", "morph"],
        top_n=7,
        min_cluster_n=10,
        min_out_n=10,
        min_pct_in=0.30,
        min_pct_diff=0.15,
        rna_min_log2fc=0.75,
        cont_min_effect=0.50,
        fdr_cutoff=0.05,
        min_shap_pos_frac=0.60,
        include_down_markers=True,
        fallback_top_n=7,
    )



    # 构建兼容旧弦图绘图函数的模态并集索引
    p_fe_idx_union = []
    for mod_idx, mod_name in enumerate(modality_names):
      mod_union = set()
      for clus_key in selected_features_dict.keys():
        mod_union.update(
            selected_features_dict[clus_key][mod_name]["indices"].tolist()
        )
      p_fe_idx_union.append(np.array(sorted(list(mod_union)), dtype=int))
    for mod_name, idx in zip(modality_names, p_fe_idx_union):
        print(
            f"{mod_name}: {len(idx)} 个绘图特征，"
            f"索引示例: {idx[:10]}"
        )

    if not any(len(idx) > 0 for idx in p_fe_idx_union):
        raise ValueError(
            "所有模态都没有可绘制特征。请检查 FDR、效应量、SHAP 阈值，"
            "或确认 pred_clusters 与 SHAP 的样本/类别顺序一致。"
        )
    # -----------------------------------------------------------------
    # 7. 绘制弦图并导出特征
    # -----------------------------------------------------------------
    all_type_features = plot_type_chord_by_major(
        shap_values=type_shap,
        feature_names_list=feature_names_list,
        p_fe_idx=p_fe_idx_union,
        clus_to_name=clus_to_name,
        major_dict=major_dict,
        save_dir=os.path.join(save_dir, f"Type_Chords_CV{cv}"),
        thres=7,
    )


    print(
        f"✓ 成功捕获 {len(all_type_features)} 个亚型的高辨识度特征字典"
        " (all_type_features)"
    )

    df_task1_features = export_task1_selected_features(all_type_features)
    print(df_task1_features.head(10))


    # =================================================================
    # 阶段二：跨模态调控 SHAP (RNA ➔ Ephys / Morph) 并绘制细分亚型弦图
    # =================================================================
    print("\n🚀 [2/2] 计算 RNA -> Ephys / Morph 跨模态调控并绘制第二套弦图...")

    trans_pairs = [("rna", "ephys", 0, 1, "0to1"), ("rna", "morph", 0, 2, "0to2")]

    target_subtypes = list(all_type_features.keys())

    for src_name, tgt_name, s_idx, t_idx, dir_str in trans_pairs:
      print(
          f"\n  ⚡ 正在处理跨模态调控: {src_name.upper()} ➔"
          f" {tgt_name.upper()}..."
      )

      # 1. 直接实例化你写好的 ExplainerWrapper_Trans
      trans_net = ExplainerWrapper_Trans(
          model, src_name=src_name, tgt_name=tgt_name
      ).to(device)
      trans_net.eval()

      # 2. 准备数据并统一设备
      src_bg = bg_tensors[s_idx].to(device)
      src_test = test_tensors_val[s_idx].to(device)

      # 3. 计算 SHAP
      trans_explainer = shap.GradientExplainer(trans_net, src_bg)
      trans_shap = trans_explainer.shap_values(src_test)

      if isinstance(trans_shap, list):
        trans_shap_arr = np.stack(trans_shap, axis=-1)
      else:
        trans_shap_arr = np.array(trans_shap)

      # 4. 保存 SHAP 缓存
      pkl_path = f"{save_dir}/trans_shap_{src_name}2{tgt_name}_cv{cv}.pkl"
      with open(pkl_path, "wb") as f:
        pickle.dump(trans_shap_arr, f)

      var_names_all = [feature_names_list[s_idx], feature_names_list[t_idx]]

      # 5. 绘制弦图并接住返回值 (cv_all, io_names_all)
      cv_all, io_names_all = feature_relevance_chord_plot(
          shap_values_0_1=trans_shap_arr,
          unique_ct=target_subtypes,
          var_names_all=var_names_all,
          all_type_features=all_type_features,
          technique=f"PatchSeq_{src_name}2{tgt_name}_cv{cv}",
          in_mod=0,
          direction=dir_str,
          thres=None,
          fontsize_names=20,
      )
      print(
          f"  ✅ {src_name.upper()} ➔ {tgt_name.upper()} 调控弦图已导出至"
          " ./feature_relevance_figures/"
      )

      # 6. 导出调控权重 CSV 表格
      export_chord_regulation_pairs(
          cv_all=cv_all,
          io_names_all=io_names_all,
          src_name=src_name,
          tgt_name=tgt_name,
          cv_idx=cv,
          save_dir=save_dir,
      )


if __name__ == '__main__':
    # =====================================================================
    # 1. 明确最佳权重的保存路径
    # =====================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    rna_path = "../dataset/PatchseqGABAergicNeuron/adata_RNA_TEM.h5ad"
    ephys_path = "../dataset/PatchseqGABAergicNeuron/adata_Ephys_TEM.h5ad"
    morph_path = "../dataset/PatchseqGABAergicNeuron/adata_Morph_TEM.h5ad"

    fine_label_key = "cell_type_TEM"
    num_clusters = 27

    save_dir = "D:\PycharmProjects\C1\Models\checkpoints_paired\patchseq"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "patchseq_best_finetuned_model.pt")


    # 3. 三模态网络架构配置 (Master Config)
    # ==========================================
    dim_rna = 1252
    dim_ephys = 68
    dim_morph = 514
    latent_dim = 64
    out_dim = 32

    master_config = {
        "modality_names": ["rna", "ephys", "morph"],
        "latent_dim": latent_dim,
        "proj_dim": 64,
        "cluster_gamma": 1.0,

        "encoder_specs": {
            "rna": {
                "input": dim_rna, "hidden": [512], "output": out_dim,
                "use_biases": [True, True], "dropouts": [0.1, 0.0],
                "use_batch_norms": [False, False], "use_layer_norms": [True, True],
                "activations": ["relu", ""]
            },
            "ephys": {
                "input": dim_ephys, "hidden": [128], "output": out_dim,
                "use_biases": [True, True], "dropouts": [0.1, 0.0],
                "use_batch_norms": [False, False], "use_layer_norms": [True, True],
                "activations": ["relu", ""]
            },
            "morph": {
                "input": dim_morph, "hidden": [128], "output": out_dim,
                "use_biases": [True, True], "dropouts": [0.1, 0.0],
                "use_batch_norms": [False, False], "use_layer_norms": [True, True],
                "activations": ["relu", ""]
            }
        },

        "decoder_specs": {
            "rna": {
                "input": out_dim, "hidden": [512], "output": dim_rna,
                "use_biases": [True, True], "dropouts": [0.0, 0.0],
                "use_batch_norms": [False, False], "use_layer_norms": [True, False],
                "activations": ["relu", ""]
            },
            "ephys": {
                "input": out_dim, "hidden": [128], "output": dim_ephys,
                "use_biases": [True, True], "dropouts": [0.0, 0.0],
                "use_batch_norms": [False, False], "use_layer_norms": [True, False],
                "activations": ["relu", ""]
            },
            "morph": {
                "input": out_dim, "hidden": [128], "output": dim_morph,
                "use_biases": [True, True], "dropouts": [0.0, 0.0],
                "use_batch_norms": [False, False], "use_layer_norms": [True, False],
                "activations": ["relu", ""]
            }
        },
    }
    # 4. 数据总线接入
    print("⏳ [Data Loading] 正在接入全量 Patch-seq 三模态数据总线...")
    cv = 0
    ind_dict_file = "../dataset/PatchseqGABAergicNeuron/cv_5_ind_dict.pkl"
    with open(ind_dict_file, "rb") as f:
        ind_dict = pickle.load(f)

    # 1. 接入数据
    data = build_patchseq_data(
        rna_input=rna_path,
        ephys_input=ephys_path,
        morph_input=morph_path,
        fine_label_key=fine_label_key,
        ind_dict=ind_dict,
        cv=cv,
        batch_size=512,
        split=False,
        device=device,
    )
    model = UnpairedClusterOTAE3Mod(
        master_config,
        num_clusters=num_clusters
    ).to(device)


    print(f"📦 [Checkpoint] 正在载入最佳 Epoch 权重: {best_model_path}")

    # 2. 安全加载 checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)

    # 3. 稳健的权重载入逻辑
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            # 如果是包装过的字典 {'model_state_dict': ..., 'epoch': ...}
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 如果直接保存的就是 model.state_dict() (即 OrderedDict)
            model.load_state_dict(checkpoint)
    elif isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        raise TypeError(f"无法识别的 Checkpoint 格式: {type(checkpoint)}")

    model.to(device)
    model.eval()
    print(f"✓ 成功载入模型权重: {best_model_path}")


    # =====================================================================
    # 2. 调用 SHAP 与弦图生成流程
    # =====================================================================
    run_patchseq_shap_pipeline(
        model=model,
        data=data,
        ind_dict=ind_dict,
        cv=cv,
        save_dir=f"./analysis_results/shap/"
    )
