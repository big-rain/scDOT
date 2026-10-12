import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from collections import defaultdict
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

def natural_sort_key(s):
  """自然排序辅助函数 (如 Lamp5-1-Uni 在 Lamp5-2-Uni 前面，不会出现 10 排在 2 前面)"""
  return [
      int(text) if text.isdigit() else text.lower()
      for text in re.split(r"(\d+)", str(s))
  ]

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

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def plot_cross_modal_regulation_scatters(
    top_pairs_csv,
    all_tensors,
    feature_names_list,
    aligned_sample_names,
    modality_names=["rna", "ephys", "morph"],
    p_threshold=0.1,
    r_threshold=0.45,
    n_cols=4,
    save_dir="./analysis_results/scatter_plots",
    fig_name="Regulation_Scatters_Significant.png",
):
  """筛选显著跨模态调控配对，绘制大图并导出每个子图的高清独立 SVG（Times New Roman 字体）。"""
  # -----------------------------------------------------------------
  # 1. 设置科研出版级全局字体为 Times New Roman
  # -----------------------------------------------------------------
  plt.rcParams["font.family"] = "serif"
  plt.rcParams["font.serif"] = [
      "Times New Roman",
      "DejaVu Serif",
      "Liberation Serif",
  ]
  plt.rcParams["mathtext.fontset"] = "stix"  # 数学公式匹配新罗马风格
  plt.rcParams["svg.fonttype"] = "none"  # 保持 SVG 文字为可编辑文本对象

  os.makedirs(save_dir, exist_ok=True)
  single_svg_dir = os.path.join(save_dir, "individual_morph")
  os.makedirs(single_svg_dir, exist_ok=True)

  df_pairs = pd.read_csv(top_pairs_csv)

  # 整理各模态数据矩阵为 DataFrame
  sample_dfs = {}
  for mod_idx, (tensor, f_names) in enumerate(
      zip(all_tensors, feature_names_list)
  ):
    m_name = modality_names[mod_idx].lower()
    mat = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
    sample_dfs[m_name] = pd.DataFrame(
        mat, columns=np.asarray(f_names).astype(str)
    )

  aligned_sample_names = np.asarray(aligned_sample_names).astype(str)

  # -----------------------------------------------------------------
  # 2. 预计算与阈值过滤 (p < p_threshold 且 |r| > r_threshold)
  # -----------------------------------------------------------------
  valid_records = []
  print(
      f"\n🔍 正在筛选调控对 (p < {p_threshold} 且 |r| > {r_threshold})..."
  )

  for idx, row in df_pairs.iterrows():
    group = str(row["Group"]).strip()
    src_mod = str(row["Source_Modality"]).lower()
    src_feat = str(row["Source_Feature"]).strip()
    tgt_mod = str(row["Target_Modality"]).lower()
    tgt_feat = str(row["Target_Feature"]).strip()
    strength = float(row["Regulation_Strength"])

    if (
        src_feat not in sample_dfs[src_mod]
        or tgt_feat not in sample_dfs[tgt_mod]
    ):
      continue

    x_vals = sample_dfs[src_mod][src_feat].values
    y_vals = sample_dfs[tgt_mod][tgt_feat].values

    target_mask = aligned_sample_names == group
    n_matched = int(np.sum(target_mask))

    if n_matched < 5:
      continue

    calc_x = x_vals[target_mask]
    calc_y = y_vals[target_mask]

    r, p_val = stats.pearsonr(calc_x, calc_y)

    if p_val < p_threshold and abs(r) > r_threshold:
      valid_records.append({
          "Group": group,
          "Source_Modality": src_mod,
          "Source_Feature": src_feat,
          "Target_Modality": tgt_mod,
          "Target_Feature": tgt_feat,
          "Regulation_Strength": strength,
          "Pearson_r": r,
          "P_value": p_val,
          "n_cells": n_matched,
      })

  df_valid = pd.DataFrame(valid_records)
  n_valid = len(df_valid)
  print(f"  🎯 筛选完成: 共找到 {n_valid} 个高显著性调控配对！")

  if n_valid == 0:
    print("⚠️ 未找到满足条件的配对，请适当放宽阈值。")
    return df_valid

  # 保存筛选清单
  valid_csv_path = os.path.join(
      save_dir, fig_name.replace(".png", "_filtered.csv")
  )
  df_valid.to_csv(valid_csv_path, index=False, encoding="utf-8-sig")

  # -----------------------------------------------------------------
  # 3. 绘图核心函数（共用于大图与单图保存）
  # -----------------------------------------------------------------
  default_colors = {
      "Sst": "#E41A1C",
      "Pvalb": "#377EB8",
      "Vip": "#4DAF4A",
      "Lamp5": "#984EA3",
      "Sncg": "#FF7F00",
  }

  def render_single_scatter(ax, row, is_standalone=False):
    group = row["Group"]
    src_mod = row["Source_Modality"]
    src_feat = row["Source_Feature"]
    tgt_mod = row["Target_Modality"]
    tgt_feat = row["Target_Feature"]
    strength = row["Regulation_Strength"]
    r = row["Pearson_r"]
    p_val = row["P_value"]
    n_matched = row["n_cells"]

    x_vals = sample_dfs[src_mod][src_feat].values
    y_vals = sample_dfs[tgt_mod][tgt_feat].values
    target_mask = aligned_sample_names == group

    cur_major = group.split("-")[0]
    theme_color = default_colors.get(cur_major, "#2b5c8f")

    # (1) 背景灰色散点
    ax.scatter(
        x_vals[~target_mask],
        y_vals[~target_mask],
        c="#d3d3d3",
        s=22 if is_standalone else 16,
        alpha=0.35,
        edgecolors="none",
        # label="Other Subtypes",
        zorder=1,
    )

    # (2) 目标高亮彩色散点
    ax.scatter(
        x_vals[target_mask],
        y_vals[target_mask],
        c=theme_color,
        s=55 if is_standalone else 45,
        alpha=0.9,
        edgecolors="white",
        linewidth=0.6,
        label=f"($n={n_matched}$)",
        zorder=3,
    )

    # (3) 亚型内线性回归拟合线
    sns.regplot(
        x=x_vals[target_mask],
        y=y_vals[target_mask],
        ax=ax,
        scatter=False,
        color=theme_color,
        line_kws={"linewidth": 2.2 if is_standalone else 1.8, "alpha": 0.9},
        ci=95,
        truncate=True,
    )

    p_str = (
        f"$p={p_val:.2e}$"
        if p_val < 0.001
        else f"$p={p_val:.3f}$"
        if p_val < 0.05
        else f"$p={p_val:.2f}$"
    )

    title_size = 11 if is_standalone else 9.5
    label_size = 10 if is_standalone else 8.5
    text_size = 8.5 if is_standalone else 7.0

    ax.set_title(
        f"{group}\n{src_feat} $\\rightarrow$ {tgt_feat}",
        fontsize=title_size,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlabel(f"{src_feat} ({src_mod.upper()})", fontsize=label_size)
    ax.set_ylabel(f"{tgt_feat} ({tgt_mod.upper()})", fontsize=label_size)

    stat_box_text = (
        f"Pearson $r = {r:+.2f}$\n{p_str}"
    )
    ax.text(
        0.05,
        0.92,
        stat_box_text,
        transform=ax.transAxes,
        fontsize=text_size,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.85,
            edgecolor="#cccccc",
            linewidth=0.5,
        ),
        zorder=4,
    )

    ax.legend(
        loc="lower right", frameon=True, fontsize=text_size, framealpha=0.8
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    sns.despine(ax=ax, top=True, right=True)

  # -----------------------------------------------------------------
  # 4. 逐一绘制并保存单独的 SVG 文件
  # -----------------------------------------------------------------
  print("💾 正在导出每个子图的独立 SVG 文件...")
  for idx, row in df_valid.iterrows():
    fig_single, ax_single = plt.subplots(figsize=(4.2, 3.8), dpi=300)
    render_single_scatter(ax_single, row, is_standalone=True)

    # 清理文件名非法字符
    safe_group = re.sub(r"[^\w\-_\.]", "_", str(row["Group"]))
    safe_src = re.sub(r"[^\w\-_\.]", "_", str(row["Source_Feature"]))
    safe_tgt = re.sub(r"[^\w\-_\.]", "_", str(row["Target_Feature"]))
    single_svg_name = f"{safe_group}_{safe_src}_to_{safe_tgt}.svg"

    fig_single.tight_layout()
    fig_single.savefig(
        os.path.join(single_svg_dir, single_svg_name),
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig_single)

  print(f"  ✓ {n_valid} 个单独 SVG 已保存至: {single_svg_dir}")

  # -----------------------------------------------------------------
  # 5. 绘制拼接大总图
  # -----------------------------------------------------------------
  n_rows = int(np.ceil(n_valid / n_cols))
  fig, axes = plt.subplots(
      n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), dpi=300
  )
  if n_valid == 1:
    axes = np.array([axes])
  axes = np.asarray(axes).flatten()

  for idx, row in df_valid.iterrows():
    render_single_scatter(axes[idx], row, is_standalone=False)

  for j in range(n_valid, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  save_path_png = os.path.join(save_dir, fig_name)
  save_path_pdf = os.path.join(save_dir, fig_name.replace(".png", ".pdf"))
  save_path_svg = os.path.join(save_dir, fig_name.replace(".png", ".svg"))

  plt.savefig(save_path_png, bbox_inches="tight")
  plt.savefig(save_path_pdf, bbox_inches="tight")
  plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
  plt.close()

  print(f"  ✅ 汇总大图已生成: {save_path_png} / .pdf / .svg")
  return df_valid


import os
import pickle
import anndata as ad
from dataloader.dataset_MET import build_patchseq_data
import numpy as np
import pandas as pd
import torch

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"🚀 Using device: {device}")

  rna_path = "../dataset/PatchseqGABAergicNeuron/adata_RNA_TEM.h5ad"
  ephys_path = "../dataset/PatchseqGABAergicNeuron/adata_Ephys_TEM.h5ad"
  morph_path = "../dataset/PatchseqGABAergicNeuron/adata_Morph_TEM.h5ad"
  fine_label_key = "cell_type_TEM"
  cv = 0
  ind_dict_file = "../dataset/PatchseqGABAergicNeuron/cv_5_ind_dict.pkl"

  with open(ind_dict_file, "rb") as f:
    ind_dict = pickle.load(f)

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

  modality_names = ["rna", "ephys", "morph"]
  all_tensors = [data["all"]["tensors"][m] for m in modality_names]

  # -------------------------------------------------------------
  # 1. 从 h5ad 文件读取特征名字与元数据
  # -------------------------------------------------------------
  adata_rna = ad.read_h5ad(rna_path)
  adata_ephys = ad.read_h5ad(ephys_path)
  adata_morph = ad.read_h5ad(morph_path)

  feature_names_list = [
      adata_rna.var_names.astype(str).tolist(),
      adata_ephys.var_names.astype(str).tolist(),
      adata_morph.var_names.astype(str).tolist(),
  ]

  # -------------------------------------------------------------
  # 2 & 3. 权威来源：从 patchseq_best_latent_z.csv 构建模型预测并对齐标签
  # -------------------------------------------------------------
  latent_csv_path = r"D:\PycharmProjects\C1\Models\checkpoints_paired\patchseq0.512\patchseq_best_latent_z.csv"
  df_latent = pd.read_csv(latent_csv_path)

  print("🧬 正在从 h5ad 元数据中提取真实生物学名称并执行标签对齐...")

  adata_obs = adata_rna.obs  # 💡 修正：直接使用 adata_rna 的 obs
  candidate_cols = [
      "subclass_label",
      "cell_type_TEM",
      "cluster",
      "fine_label",
      "subclass",
      "true_label",
  ]

  target_fine_col = None
  for col in candidate_cols:
    if col in adata_obs.columns:
      sample_val = str(adata_obs[col].dropna().iloc[0])
      if not sample_val.isdigit():  # 优先锁定包含文字的生物学名称列
        target_fine_col = col
        break

  if target_fine_col is None:
    for col in candidate_cols:
      if col in adata_obs.columns:
        target_fine_col = col
        break

  if target_fine_col is None:
    raise KeyError(
        f"未在 adata.obs 中找到细分标签列！当前可用列: {list(adata_obs.columns)}"
    )

  print(f"  ✓ 锁定 h5ad 细分名称列: '{target_fine_col}'")
  true_fine_names = adata_obs[target_fine_col].astype(str).values

  # 提取模型预测簇 (优先找 kmeans_pred，若没有则自动兼容)
  pred_col = (
      "kmeans_pred"
      if "kmeans_pred" in df_latent.columns
      else ("cluster" if "cluster" in df_latent.columns else "pred")
  )
  pred_clusters = df_latent[pred_col].values

  # 提取真实唯一类别并编码
  fine_classes = np.unique(true_fine_names)
  fine_to_idx = {cls_name: i for i, cls_name in enumerate(fine_classes)}
  fine_encoded = np.array([fine_to_idx[x] for x in true_fine_names])

  # 执行簇对齐
  clus_to_name, major_dict, aligned_sample_names = align_clusters_to_metadata(
      pred_clusters=pred_clusters,
      fine_encoded=fine_encoded,
      fine_classes=fine_classes,
  )

  # 🚨 样本对齐安全性检查
  num_samples_tensor = all_tensors[0].shape[0]
  num_samples_pred = len(aligned_sample_names)
  assert num_samples_tensor == num_samples_pred, (
      f"❌ 样本数不匹配！all_tensors 包含 {num_samples_tensor} 个细胞，"
      f"但 latent_z.csv 包含 {num_samples_pred} 行！"
  )

  print(
      f"✅ 簇对齐映射成功 (共 {len(clus_to_name)} 个模型簇, {num_samples_pred} 个细胞):"
  )
  for k, v in sorted(
      clus_to_name.items(),
      key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
  ):
    print(f"    Cluster {k}  -->  {v}")

  # 4. 开始绘制 Task 3 散点图
  save_dir = r"./analysis_results/"
  top1_csv_path = r"D:\PycharmProjects\C1\notebook\analysis_results\shap\Task1_Regulation_All_Pairs_rna2morph_cv0.csv"
  src_name, tgt_name = "rna", "morphs"

  plot_cross_modal_regulation_scatters(
      top_pairs_csv=top1_csv_path,
      all_tensors=all_tensors,
      feature_names_list=feature_names_list,
      aligned_sample_names=aligned_sample_names,  # 👈 核心修正：传 aligned_sample_names
      modality_names=modality_names,
      n_cols=4,
      save_dir=os.path.join(save_dir, "scatters"),
      fig_name=f"Regulation_Scatter_{src_name}2{tgt_name}_cv{cv}.png",
  )

