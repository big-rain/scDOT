from collections import defaultdict
import json
import re
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,  # <-- 引入 FMI 计算函数
)

# =====================================================================
# 全局样式配置：Times New Roman + SVG 文本可编辑核心配置
# =====================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

# 确保导出 SVG 时为原生可编辑文本
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]

# =====================================================================
# 1. 加载元数据与聚类结果
# =====================================================================
meta_path = (
    r"D:\PycharmProjects\C1\dataset\PatchseqGABAergicNeuron\patchseq_metadata.json"
)
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

idx_to_name = {int(k): v for k, v in meta["index_to_fine"].items()}

csv_path = r"D:\PycharmProjects\C1\Models\check_E\patchseq\patchseq_best_latent_z.csv"
df = pd.read_csv(csv_path)

df["true_label_name"] = df["true_label"].astype(int).map(idx_to_name)
df["raw_pred_cluster"] = [f"TE-{int(x):02d}" for x in df["kmeans_pred"]]
df["true_coarse"] = df["true_label_name"].apply(lambda x: x.split("-")[0])

# =====================================================================
# 2. 矩阵排序与自动对齐映射
# =====================================================================
ct_temp = pd.crosstab(df["raw_pred_cluster"], df["true_label_name"])
ordered_fine_classes = sorted(
    [c for c in meta["fine_classes"] if c in ct_temp.columns],
    key=natural_sort_key,
)
ct_temp = ct_temp.reindex(columns=ordered_fine_classes, fill_value=0)

col_idx_map = {col: idx for idx, col in enumerate(ct_temp.columns)}
col_weights = np.arange(ct_temp.shape[1])

def get_sort_key(row_name):
    primary_col = ct_temp.loc[row_name].idxmax()
    return (
        col_idx_map[primary_col],
        np.average(col_weights, weights=ct_temp.loc[row_name].values),
    )

sorted_raw_rows = sorted(ct_temp.index, key=get_sort_key)

coarse_counters = defaultdict(int)
cluster_rename_map = {}
for raw_id in sorted_raw_rows:
    coarse = ct_temp.loc[raw_id].idxmax().split("-")[0]
    coarse_counters[coarse] += 1
    cluster_rename_map[raw_id] = f"{coarse}-MET-{coarse_counters[coarse]}"

df["pred_cluster"] = df["raw_pred_cluster"].map(cluster_rename_map)
row_order = [cluster_rename_map[r] for r in sorted_raw_rows]

# 细类与粗类交叉矩阵
ct_fine = pd.crosstab(df["pred_cluster"], df["true_label_name"]).reindex(
    columns=ordered_fine_classes, index=row_order, fill_value=0
)

coarse_subclasses = ["Lamp5", "Pvalb", "Sncg", "Sst", "Vip"]
ordered_coarse_classes = [
    c for c in coarse_subclasses if c in df["true_coarse"].unique()
]
ct_coarse = pd.crosstab(df["pred_cluster"], df["true_coarse"]).reindex(
    columns=ordered_coarse_classes, index=row_order, fill_value=0
)

# 行归一化 (Row-normalized)
ct_coarse_norm = ct_coarse.div(ct_coarse.sum(axis=1), axis=0).fillna(0)
ct_fine_norm = ct_fine.div(ct_fine.sum(axis=1), axis=0).fillna(0)

# =====================================================================
# 3. 严谨计算 ARI / NMI / FMI
# =====================================================================
# --- Fine-grained 细类指标 ---
ari_fine = adjusted_rand_score(df["true_label_name"], df["pred_cluster"])
nmi_fine = normalized_mutual_info_score(
    df["true_label_name"], df["pred_cluster"]
)
fmi_fine = fowlkes_mallows_score(
    df["true_label_name"], df["pred_cluster"]
)

# --- Subclass 粗类指标 ---
df["pred_coarse"] = df["pred_cluster"].apply(lambda x: x.split("-")[0])
ari_coarse = adjusted_rand_score(df["true_coarse"], df["pred_coarse"])
nmi_coarse = normalized_mutual_info_score(
    df["true_coarse"], df["pred_coarse"]
)
fmi_coarse = fowlkes_mallows_score(
    df["true_coarse"], df["pred_coarse"]
)

print(
    f">>> Subclass:     ARI = {ari_coarse:.4f}, NMI = {nmi_coarse:.4f}, FMI = {fmi_coarse:.4f}"
)
print(
    f">>> Fine-grained: ARI = {ari_fine:.4f}, NMI = {nmi_fine:.4f}, FMI = {fmi_fine:.4f}"
)

# =====================================================================
# 4. 紧凑组合热图
# =====================================================================
fig = plt.figure(figsize=(10.5, 5.6))

ncols_coarse = ct_coarse_norm.shape[1]
ncols_fine = ct_fine_norm.shape[1]

# wspace 极小保持紧凑
gs = gridspec.GridSpec(
    1, 3, width_ratios=[ncols_coarse, ncols_fine, 0.6], wspace=0.03
)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])

cmap_theme = "Blues"

# --- Panel 1: Subclass (粗类，标题加入 FMI) ---
sns.heatmap(
    ct_coarse_norm,
    ax=ax1,
    cmap=cmap_theme,
    vmin=0.0,
    vmax=1.0,
    annot=False,
    linewidths=0.0,
    cbar=False,
)
ax1.set_xticklabels(
    ct_coarse_norm.columns, rotation=90, ha="center", va="top"
)
ax1.set_yticklabels(ct_coarse_norm.index, rotation=0)
ax1.set_title(
    f"Subclass\n(ARI={ari_coarse:.2f}, NMI={nmi_coarse:.2f}, FMI={fmi_coarse:.2f})",
    fontsize=10.5,
    pad=6,
)

# --- Panel 2: Fine-grained (细类，标题加入 FMI) ---
sns.heatmap(
    ct_fine_norm,
    ax=ax2,
    cmap=cmap_theme,
    vmin=0.0,
    vmax=1.0,
    annot=False,
    linewidths=0.0,
    cbar_ax=cax,
    cbar_kws={"label": "Fraction"},
)
ax2.set_xticklabels(
    ct_fine_norm.columns, rotation=90, ha="center", va="top"
)
ax2.set_yticks([])  # 共享左侧 Y 轴刻度
ax2.set_title(
    f"Fine-grained\n(ARI={ari_fine:.2f}, NMI={nmi_fine:.2f}, FMI={fmi_fine:.2f})",
    fontsize=10.5,
    pad=6,
)

# --- 清理所有边框、标签与 Ticks ---
for ax in [ax1, ax2]:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Colorbar 微调
cax.tick_params(labelsize=10, length=2)
cax.yaxis.label.set_size(11)

plt.tight_layout()

# 保存为矢量 SVG
save_svg_path = "nature_compact_alignment_heatmap.svg"
plt.savefig(
    save_svg_path,
    format="svg",
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()

print(f"✓ 修复完成！SVG 已保存至: {save_svg_path}")