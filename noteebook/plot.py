from collections import defaultdict
import json
import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

# =====================================================================
# 全局排版配置：Times New Roman + SVG 原生矢量可编辑
# =====================================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "none"  # AI / Inkscape 中可自由编辑文本
plt.rcParams["pdf.fonttype"] = 42


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


# =====================================================================
# 1. 加载元数据与数据
# =====================================================================
meta_path = (
    r"D:\PycharmProjects\C1\dataset\PatchseqGABAergicNeuron\patchseq_metadata.json"
)
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

idx_to_name = {int(k): v for k, v in meta["index_to_fine"].items()}

csv_path = r"D:\PycharmProjects\C1\Models\checkpoints_paired\patchseq0.512\patchseq_best_latent_z.csv"
df = pd.read_csv(csv_path)

df["true_label_name"] = df["true_label"].astype(int).map(idx_to_name)
df["true_coarse"] = df["true_label_name"].apply(lambda x: x.split("-")[0])
df["raw_pred_cluster"] = [f"TE-{int(x):02d}" for x in df["kmeans_pred"]]

# =====================================================================
# 2. 预测簇自动匹配重命名
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
    cluster_rename_map[raw_id] = f"{coarse}-{coarse_counters[coarse]}-Uni"

df["pred_fine"] = df["raw_pred_cluster"].map(cluster_rename_map)
df["pred_coarse"] = df["pred_fine"].apply(lambda x: x.split("-")[0])

# =====================================================================
# 3. 提取潜变量并计算 UMAP
# =====================================================================
feature_cols = [
    c
    for c in df.columns
    if c.startswith("z")
    or c.startswith("latent")
    or c.startswith("dim")
    or c.isdigit()
]
X = df[feature_cols].values

reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
embedding = reducer.fit_transform(X)
df["umap_1"] = embedding[:, 0]
df["umap_2"] = embedding[:, 1]

# =====================================================================
# 4. 调色盘配置
# =====================================================================
coarse_palette = {
    "Lamp5": "#7B3294",  # 紫色
    "Pvalb": "#2B83BA",  # 蓝色
    "Sncg": "#ABdda4",  # 浅青绿
    "Sst": "#FDAE61",  # 暖橙黄
    "Vip": "#D7191C",  # 鲜红
}

coarse_gradients = {
    "Lamp5": ["#542788", "#998EC3"],
    "Pvalb": ["#084594", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1"],
    "Sncg": ["#238B45", "#74C476", "#A1D99B", "#C7E9C0"],
    "Sst": [
        "#8C510A",
        "#BF812D",
        "#DFC27D",
        "#F6E8C3",
        "#C7EAE5",
        "#80CDC1",
        "#35978F",
        "#01665E",
        "#E08214",
        "#FDB863",
        "#FEE0B6",
    ],
    "Vip": ["#67000D", "#A50F15", "#CB181D", "#EF3B2C", "#FB6A4A", "#FC9272"],
}


def build_fine_palette(unique_fines):
    fine_palette = {}
    grouped = defaultdict(list)
    for f_name in sorted(unique_fines, key=natural_sort_key):
        c = f_name.split("-")[0]
        grouped[c].append(f_name)

    for c, f_list in grouped.items():
        grad_colors = coarse_gradients.get(
            c, [coarse_palette.get(c, "#333333")]
        )
        cmap_sub = mcolors.LinearSegmentedColormap.from_list(
            f"sub_{c}", grad_colors
        )
        if len(f_list) == 1:
            fine_palette[f_list[0]] = coarse_palette.get(c, grad_colors[0])
        else:
            for idx, item in enumerate(f_list):
                val = (
                    idx / (len(f_list) - 1)
                    if len(f_list) > 1
                    else 0.5
                )
                fine_palette[item] = mcolors.to_hex(cmap_sub(val))
    return fine_palette


fine_palette_true = build_fine_palette(df["true_label_name"].unique())
fine_palette_pred = build_fine_palette(df["pred_fine"].unique())


# =====================================================================
# 5. 纯净散点绘制与 SVG 导出函数
# =====================================================================
# =====================================================================
# 5. 纯净散点绘制与 SVG 导出函数（修复挤压变形问题）
# =====================================================================
def plot_and_save_single_umap(
    hue_col,
    palette,
    filename,
    is_fine=False,
    main_size=3.8,  # 散点图主框的固定物理尺寸 (英寸)，保证 4 张图严格等大
):
    # 根据图例大小，动态为图例预留画布右侧宽度，避免挤压主图
    legend_w = 3.8 if is_fine else 2.2
    fig_w = main_size + legend_w
    fig_h = (
        main_size + 0.4
    )  # 细类标签行数较多时，适度预留垂直高度：max(main_size + 0.4, 4.5) if is_fine else main_size + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # 关键：使用绝对比例固定绘图主轴，不受图例字号影响！
    # [left, bottom, width, height] (归一化坐标)
    ax_left = 0.05
    ax_bottom = 0.05
    ax_w = main_size / fig_w
    ax_h = main_size / fig_h
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])

    # 散点图强制保持 1:1 正方形比例，绝不被拉伸压扁
    ax.set_aspect("equal", adjustable="datalim")

    # 纯净散点绘制
    labels = sorted(df[hue_col].unique(), key=natural_sort_key)
    for lbl in labels:
        sub_df = df[df[hue_col] == lbl]
        ax.scatter(
            sub_df["umap_1"],
            sub_df["umap_2"],
            c=palette[lbl],
            label=lbl,
            s=22,
            alpha=0.9,
            edgecolors="none",
        )

    # 黑色细边框（学术顶刊标准）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)
        spine.set_visible(True)

    # 图例配置：紧凑行距，防止文字互相遮挡或超高
    if is_fine:
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
            ncol=2,
            fontsize=20,  # 细类 20+ 项，建议 12-13pt 最协调；如必须 15pt 可保持
            labelspacing=0.25,  # 紧凑垂直行距，防止太长
            columnspacing=0.6,  # 两列之间的水平间距
            handletextpad=0.2,
            markerscale=1.2,
            borderpad=0,
        )
    else:
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
            ncol=1,
            fontsize=20,
            labelspacing=0.4,
            handletextpad=0.3,
            markerscale=1.4,
            borderpad=0,
        )

    # 注意：千万不要调用 plt.tight_layout()！
    # bbox_inches="tight" 会自动裁剪外部多余留白，同时 100% 保持 ax 尺寸不变
    plt.savefig(filename, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"✓ 已生成无压缩矢量图: {filename}")


# =====================================================================
# 6. 生成 4 张独立的 SVG 子图（主框统一为 3.8 x 3.8 英寸正方形）
# =====================================================================
# 1. 真值粗类 (b-i)
plot_and_save_single_umap(
    hue_col="true_coarse",
    palette=coarse_palette,
    filename="umap_true_coarse.svg",
    is_fine=False,
    main_size=3.8,
)

# 2. 真值细类 (b-ii)
plot_and_save_single_umap(
    hue_col="true_label_name",
    palette=fine_palette_true,
    filename="umap_true_fine.svg",
    is_fine=True,
    main_size=3.8,
)

# 3. 预测粗类 (c-i)
plot_and_save_single_umap(
    hue_col="pred_coarse",
    palette=coarse_palette,
    filename="umap_pred_coarse.svg",
    is_fine=False,
    main_size=3.8,
)

# 4. 预测细类 (c-ii)
plot_and_save_single_umap(
    hue_col="pred_fine",
    palette=fine_palette_pred,
    filename="umap_pred_fine.svg",
    is_fine=True,
    main_size=3.8,
)

print("\n🎉 4 张等大、正方形、绝不被压缩的 UMAP 全部保存完毕！")

