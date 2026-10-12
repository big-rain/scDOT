import scanpy as sc

import umap
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import re
import matplotlib.pyplot as plt

# 全局配置为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"  # 保证导出 SVG 时文字在 AI 中可编辑

# ==============================================================================
# 1. 辅助函数：将聚类结果与 AnnData 对齐并注入
# ==============================================================================
def load_and_sync_labels(latent_csv_path, adatas_all):
    """
    读取 latent_z.csv，并将预测聚类标签和真实标签注入到每一个 adata.obs 中
    """
    if not os.path.exists(latent_csv_path):
        raise FileNotFoundError(f"未找到特征表示文件: {latent_csv_path}")

    print(f"📖 正在加载表征与聚类文件: {latent_csv_path}")
    df_latent = pd.read_csv(latent_csv_path)
    print(f"数据总行数: {len(df_latent)}, 包含列: {list(df_latent.columns[:5])} ... {list(df_latent.columns[-5:])}")

    # 提取聚类标签（保证转换为带前缀的字符串形式，便于画图展示，如 'Cluster_0'）
    kmeans_labels = [f"Cluster_{int(c)}" for c in df_latent['kmeans_pred']]
    true_labels = df_latent['true_label'].astype(str).tolist()

    # 注入到所有 AnnData 对象
    for i, adata in enumerate(adatas_all):
        if adata is not None:
            # 确认样本量对齐
            assert len(adata) == len(
                df_latent), f"模态 {i} 的样本数 ({len(adata)}) 与 Latent 表格行数 ({len(df_latent)}) 不一致！"
            adata.obs['kmeans_pred'] = kmeans_labels
            adata.obs['true_label'] = true_labels

    print("✅ 聚类标签成功注入 adatas_all！")
    return df_latent


# ==============================================================================
# 2. 核心函数：绘制特征箱线图面板 (Task 2)
# ==============================================================================


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
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_modality_boxplots_from_adata(
    adata,
    selected_features=None,
    modality_name="RNA",
    group_col="pred_sub",
    clus_to_name=None,
    ncols=4,
    save_dir="./analysis_results",
    dpi=300,
):
  """绘制模态特征箱线图（支持容错模糊匹配、缺失排查、字体全局统一）"""
  # 设置全局字体为 Times New Roman
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["svg.fonttype"] = "none"

  os.makedirs(save_dir, exist_ok=True)
  single_plots_dir = os.path.join(save_dir, f"Single_Boxplots_{modality_name}")
  os.makedirs(single_plots_dir, exist_ok=True)

  # ----------------------------------------------------
  # 1. 彻底排查并对齐特征 (含大小写容错映射)
  # ----------------------------------------------------
  if selected_features is None:
    valid_features = list(adata.var_names)
  else:
    # 建立 var_names 的【纯小写 -> 真实名称】映射字典
    var_map_lower = {str(v).strip().lower(): str(v) for v in adata.var_names}

    valid_features = []
    missing_features = []
    seen = set()

    for feat in selected_features:
      feat_clean = str(feat).strip()
      # 1. 精确匹配
      if feat_clean in adata.var_names:
        matched_feat = feat_clean
      # 2. 忽略大小写匹配
      elif feat_clean.lower() in var_map_lower:
        matched_feat = var_map_lower[feat_clean.lower()]
      else:
        matched_feat = None

      if matched_feat and (matched_feat not in seen):
        seen.add(matched_feat)
        valid_features.append(matched_feat)
      elif not matched_feat:
        missing_features.append(feat)

    # 打印排查真相
    if missing_features:
      print(f"\n❌ [{modality_name}] 以下 {len(missing_features)} 个特征未在 adata.var_names 中找到 (被过滤跳过):")
      for mf in missing_features:
        # 顺便检查是否不小心放进了 adata.obs
        in_obs_hint = (
            " (⚠️ 注意: 该特征存在于 adata.obs 中!)"
            if mf in adata.obs.columns
            else ""
        )
        print(f"   • '{mf}'{in_obs_hint}")
      print("-" * 60)

  n_feats = len(valid_features)
  if n_feats == 0:
    print(f"⚠️ [{modality_name}] 有效特征数为 0，终止绘图！")
    return

  # ----------------------------------------------------
  # 2. 标签映射
  # ----------------------------------------------------
  plot_obs_group = adata.obs[group_col].copy().astype(str)
  if clus_to_name is not None:
    id_to_name = {}
    for k, v in clus_to_name.items():
      nums = re.findall(r"\d+", str(k))
      if nums:
        id_to_name[int(nums[0])] = str(v)

    def convert_label(val):
      nums = re.findall(r"\d+", str(val))
      if nums:
        clus_id = int(nums[0])
        return id_to_name.get(clus_id, str(val))
      return str(val)

    plot_obs_group = plot_obs_group.apply(convert_label)

  # 3. 排序
  def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]

  unique_groups = plot_obs_group.dropna().unique()
  ordered_groups = sorted(unique_groups, key=natural_sort_key)

  # 4. 调色板
  major_types = [str(g).split("-")[0] for g in ordered_groups]
  unique_majors = sorted(list(set(major_types)))
  palette_base = sns.color_palette("tab10", n_colors=len(unique_majors))
  major_to_color = dict(zip(unique_majors, palette_base))
  group_palette = {
      g: major_to_color[str(g).split("-")[0]] for g in ordered_groups
  }

  print(
      f"📊 [{modality_name}] 开始绘图：成功匹配 {n_feats} 个特征，亚型数:"
      f" {len(ordered_groups)}"
  )

  # ----------------------------------------------------
  # 任务 A：保存单图
  # ----------------------------------------------------
  for feat in valid_features:
    data_col = adata[:, feat].X
    feat_values = (
        data_col.toarray().flatten()
        if hasattr(data_col, "toarray")
        else np.asarray(data_col).flatten()
    )
    plot_df = pd.DataFrame(
        {"Group": plot_obs_group.values, "Value": feat_values}
    )

    fig_single, ax_single = plt.subplots(figsize=(7.5, 4.5))

    sns.boxplot(
        data=plot_df,
        x="Group",
        y="Value",
        order=ordered_groups,
        ax=ax_single,
        palette=group_palette,
        fliersize=0,
        boxprops=dict(alpha=0.65),
        linewidth=1.2,
    )
    sns.stripplot(
        data=plot_df,
        x="Group",
        y="Value",
        order=ordered_groups,
        ax=ax_single,
        color="black",
        alpha=0.3,
        size=3.0,
        jitter=0.2,
    )

    ax_single.set_title(f"{feat}", fontsize=13, fontweight="bold", pad=8)
    ax_single.set_ylabel("Raw / Normalized Value", fontsize=11)
    ax_single.set_xlabel("")
    ax_single.tick_params(axis="x", rotation=45, labelsize=9)
    ax_single.spines["top"].set_visible(False)
    ax_single.spines["right"].set_visible(False)
    plt.tight_layout()

    safe_feat_name = (
        str(feat).replace("/", "_").replace("\\", "_").replace(" ", "_")
    )
    single_path = os.path.join(
        single_plots_dir, f"{modality_name}_{safe_feat_name}.svg"
    )
    fig_single.savefig(single_path, bbox_inches="tight", format="svg")
    plt.close(fig_single)

  print(f"  ✓ 单图已输出至: {single_plots_dir}")

  # ----------------------------------------------------
  # 任务 B：汇总大图
  # ----------------------------------------------------
  nrows = int(np.ceil(n_feats / ncols))
  fig_all, axes_all = plt.subplots(
      nrows=nrows,
      ncols=ncols,
      figsize=(ncols * 4.8, nrows * 3.5),
      squeeze=False,
  )
  axes_flat = axes_all.flatten()

  for i, feat in enumerate(valid_features):
    ax = axes_flat[i]
    data_col = adata[:, feat].X
    feat_values = (
        data_col.toarray().flatten()
        if hasattr(data_col, "toarray")
        else np.asarray(data_col).flatten()
    )
    plot_df = pd.DataFrame(
        {"Group": plot_obs_group.values, "Value": feat_values}
    )

    sns.boxplot(
        data=plot_df,
        x="Group",
        y="Value",
        order=ordered_groups,
        ax=ax,
        palette=group_palette,
        fliersize=0,
        boxprops=dict(alpha=0.65),
        linewidth=1.0,
    )
    sns.stripplot(
        data=plot_df,
        x="Group",
        y="Value",
        order=ordered_groups,
        ax=ax,
        color="black",
        alpha=0.25,
        size=2.0,
        jitter=0.2,
    )

    ax.set_title(f"{feat}", fontsize=11, fontweight="bold", pad=5)
    ax.set_ylabel("Raw / Norm", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

  for j in range(n_feats, len(axes_flat)):
    fig_all.delaxes(axes_flat[j])

  plt.tight_layout()
  all_path = os.path.join(save_dir, f"Task2_Boxplot_{modality_name}_All.svg")
  fig_all.savefig(all_path, bbox_inches="tight", format="svg")
  plt.close(fig_all)
  print(f"  ✓ 汇总大图已保存至: {all_path}\n")
# ==============================================================================
# 3. 增强功能：用 df_latent 直接绘制发表级 UMAP (双配色对比)
# ==============================================================================
def plot_latent_umap(df_latent, save_dir='./analysis_results'):
    """
    使用提取的 latent_z 矩阵生成 UMAP，并画出 15 聚类 vs 原始真值亚型
    """
    os.makedirs(save_dir, exist_ok=True)
    # 排除标签列，其余列为 Latent Embedding 维度
    feat_cols = [c for c in df_latent.columns if c not in ['kmeans_pred', 'true_label', 'sample_id', 'cell_id']]
    X_latent = df_latent[feat_cols].values

    print(f"🗺️ 正在根据多模态潜在空间 ({X_latent.shape[1]} 维) 计算 UMAP...")
    reducer = umap.UMAP(n_neighbors=25, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(X_latent)

    df_plot = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    df_plot['kmeans_pred'] = [f"Cluster_{int(c)}" for c in df_latent['kmeans_pred']]
    df_plot['true_label'] = df_latent['true_label'].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图 1：15 个跨模态自学习聚类
    sns.scatterplot(
        data=df_plot, x='UMAP_1', y='UMAP_2', hue='kmeans_pred',
        palette='tab20', s=16, alpha=0.8, ax=axes[0], legend='full'
    )
    axes[0].set_title("A. Latent Space (Colored by Learned 15 Clusters)", fontsize=13, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1, fontsize=8)
    axes[0].axis('off')

    # 子图 2：已知先验亚型标签
    sns.scatterplot(
        data=df_plot, x='UMAP_1', y='UMAP_2', hue='true_label',
        palette='tab20', s=16, alpha=0.8, ax=axes[1], legend=False
    )
    axes[1].set_title("B. Latent Space (Colored by Prior True Labels)", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "Latent_UMAP_Comparison.pdf")
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ UMAP 对照图已保存至: {save_path}")


import os
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def main():
  # -----------------------------------------------------------------
  # 1. 路径配置
  # -----------------------------------------------------------------
  rna_path = "../dataset/PatchseqGABAergicNeuron/adata_RNA_TEM.h5ad"
  ephys_path = "../dataset/PatchseqGABAergicNeuron/adata_Ephys_TEM.h5ad"
  morph_path = "../dataset/PatchseqGABAergicNeuron/adata_Morph_TEM.h5ad"

  latent_csv_path = r"D:\PycharmProjects\C1\Models\checkpoints_paired\patchseq0.512\patchseq_best_latent_z.csv"
  # latent_csv_path = r"D:/PycharmProjects/C1/Models/checkpoints_paired/patchseq/patchseq_best_latent_z.csv"
  features_csv_path = "./analysis_results/Task1_Selected_Features_Summary.csv"
  save_dir = "./analysis_results"

  # -----------------------------------------------------------------
  # 2. 载入原始 AnnData 并同步潜空间预测标签
  # -----------------------------------------------------------------
  print("📂 正在读取三模态 AnnData 数据...")
  adatas_all = [
      sc.read_h5ad(rna_path),
      sc.read_h5ad(ephys_path),
      sc.read_h5ad(morph_path),
  ]

  print("🔗 正在同步 Latent 表征与标签元数据...")
  df_latent = load_and_sync_labels(latent_csv_path, adatas_all)

  # -----------------------------------------------------------------
  # 3. 提取真实细分生物学名称并执行簇对齐 (align_clusters_to_metadata)
  # -----------------------------------------------------------------
  print("🧬 正在从 h5ad 元数据中提取真实生物学名称并执行标签对齐...")

  adata_obs = adatas_all[0].obs
  candidate_cols = [
      "subclass_label",
      "cluster",
      "cell_type_TEM",
      "fine_label",
      "subclass",
      "true_label",
  ]

  target_fine_col = None
  for col in candidate_cols:
    if col in adata_obs.columns:
      sample_val = str(adata_obs[col].dropna().iloc[0])
      # 优先锁定包含文字的生物学名称列
      if not sample_val.isdigit():
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
  pred_clusters = df_latent["kmeans_pred"].values

  # 提取真实唯一类别并编码
  fine_classes = np.unique(true_fine_names)
  fine_to_idx = {cls_name: i for i, cls_name in enumerate(fine_classes)}
  fine_encoded = np.array([fine_to_idx[x] for x in true_fine_names])

  # 执行簇对齐并打印映射预览
  clus_to_name, major_dict, aligned_sample_names = align_clusters_to_metadata(
      pred_clusters=pred_clusters,
      fine_encoded=fine_encoded,
      fine_classes=fine_classes,
  )

  print("  ✓ 簇对齐映射关系预览:")
  for k, v in sorted(
      clus_to_name.items(),
      key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
  ):
    print(f"    Cluster {k}  -->  {v}")

  # -----------------------------------------------------------------
  # 4. 读取 Task 1 导出的特征，并为每个模态画箱线图
  # -----------------------------------------------------------------
  print(f"\n📊 正在读取选定特征列表: {features_csv_path}")
  df_features = pd.read_csv(features_csv_path)

  # 列名兼容（支持 Importance_Rank 或 Rank）
  rank_col = "Importance_Rank" if "Importance_Rank" in df_features.columns else (
      "Rank" if "Rank" in df_features.columns else None
  )
  feat_col = (
      "Feature_Name"
      if "Feature_Name" in df_features.columns
      else df_features.columns[3]
  )
  mod_col = (
      "Modality" if "Modality" in df_features.columns else df_features.columns[1]
  )

  modalities = [("RNA", 0), ("Ephys", 1), ("Morphology", 2)]

  for mod_name, mod_idx in modalities:
    adata_mod = adatas_all[mod_idx]

    # 模态过滤容错（兼容 MORPH 与 MORPHOLOGY）
    mod_mask = df_features[mod_col].astype(str).str.upper().str.startswith(
        mod_name[:4].upper()
    )
    sub_df = df_features[mod_mask]

    if rank_col is not None:
      sub_df = sub_df.sort_values(by=rank_col)

    selected_feats = sub_df[feat_col].drop_duplicates().tolist()

    print(
        f"\n🎨 [{mod_name}] 正在绘制箱线图 (共 {len(selected_feats)} 个候选特征)..."
    )

    plot_modality_boxplots_from_adata(
        adata=adata_mod,
        selected_features=selected_feats,
        modality_name=mod_name,
        group_col="kmeans_pred",
        clus_to_name=clus_to_name,
        ncols=4,
        save_dir=os.path.join(save_dir, "Boxplots_Aligned"),
        dpi=300,
    )

  print(
      f"\n✨ [全部完成] 所有模态箱线图已成功对齐并保存至:"
      f" {os.path.join(save_dir, 'Boxplots_Aligned')}"
  )

if __name__ == "__main__":
  main()

