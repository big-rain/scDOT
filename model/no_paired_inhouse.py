


from MLP import MLP

import sys
import os
import random
import numpy as np

from torch.optim import AdamW
from tqdm import tqdm

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import connected_components

# from functions.loss import HighThroughputPartialOT

from dataloader.data_pbmc import load_real_folder_pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 若你追求完全可复现，可以开启下面两行
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






@torch.no_grad()
def extract_z_and_labels_from_dataset(
    model,
    loader,
    device,
    save_csv_path=None,
    normalize_z=True,
):

    model.eval()
    dataset = loader.dataset

    z_rna_list, z_atac_list = [], []
    y_rna_list, y_atac_list = [], []
    barcode_rna_list, barcode_atac_list = [], []

    for batch in loader:
        batch_dict = {
            "rna": batch["rna"].to(device, non_blocking=True),
            "atac": batch["atac"].to(device, non_blocking=True),
        }

        out = model(batch_dict)

        z_rna = out["z_rna"]
        z_atac = out["z_atac"]

        # 与 cosine OT 使用的球面几何保持一致。
        if normalize_z:
            z_rna = F.normalize(z_rna, dim=-1)
            z_atac = F.normalize(z_atac, dim=-1)

        z_rna_list.append(z_rna.cpu().numpy())
        z_atac_list.append(z_atac.cpu().numpy())

        y_rna_list.append(batch["label"].cpu().numpy())

        atac_indices = batch["atac_idx"].cpu().numpy()
        y_atac_list.append(dataset.labels[atac_indices].cpu().numpy())

        # 默认 DataLoader 会将字符串 barcode collate 成 list/tuple。
        barcode_rna_list.extend([str(x) for x in batch["rna_barcode"]])
        barcode_atac_list.extend([str(x) for x in batch["atac_barcode"]])

    z_rna_all = np.concatenate(z_rna_list, axis=0)
    z_atac_all = np.concatenate(z_atac_list, axis=0)

    y_rna_all = np.concatenate(y_rna_list, axis=0)
    y_atac_all = np.concatenate(y_atac_list, axis=0)

    z_joint = np.concatenate([z_rna_all, z_atac_all], axis=0)
    modality = np.array(
        ["RNA"] * len(z_rna_all) + ["ATAC"] * len(z_atac_all),
        dtype=str,
    )
    cell_type_id = np.concatenate([y_rna_all, y_atac_all], axis=0)
    barcode = np.asarray(barcode_rna_list + barcode_atac_list, dtype=str)

    # 数字标签映射为实际类别名称。
    if hasattr(dataset, "label_names") and dataset.label_names is not None:
        label_names = np.asarray(dataset.label_names, dtype=str)

        valid_label = (
            (cell_type_id >= 0) &
            (cell_type_id < len(label_names))
        )

        cell_type = cell_type_id.astype(str)
        cell_type[valid_label] = label_names[cell_type_id[valid_label]]
    else:
        cell_type = cell_type_id.astype(str)

    if save_csv_path is not None:
        output_dir = os.path.dirname(save_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        latent_columns = [f"z_{i}" for i in range(z_joint.shape[1])]

        latent_df = pd.DataFrame(z_joint, columns=latent_columns)
        metadata_df = pd.DataFrame({
            "barcode": barcode,
            "modality": modality,
            "cell_type": cell_type,
        })

        # 明确保证 barcode 第一列，z 随后，最后为模态和类别。
        export_df = pd.concat(
            [
                metadata_df[["barcode"]],
                latent_df,
                metadata_df[["modality", "cell_type"]],
            ],
            axis=1,
        )

        export_df.to_csv(save_csv_path, index=False, float_format="%.8f")
        print(f"Latent features saved: {save_csv_path}")
        print(
            f"Rows={len(export_df)}, latent_dim={z_joint.shape[1]}, "
            f"RNA={len(z_rna_all)}, ATAC={len(z_atac_all)}"
        )

    return z_joint, modality, cell_type


def plot_best_joint_z_umap(model, loader, device, save_dir, epoch, joint_z_ari):
    """
    绘制 Joint Z 空间的 UMAP 图（包含 Modality 和 Cell Type 两张）
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🎨 [Epoch {epoch}] 正在提取 Latent z 并计算 UMAP (Joint z ARI = {joint_z_ari:.4f})...")

    # 1. 提取数据

    z_joint, modality, cell_type = extract_z_and_labels_from_dataset(
        model=model,
        loader=loader,
        device=device,
        normalize_z=True,
        save_csv_path=os.path.join(
            save_dir,
            f"best_joint_latent_epoch_{epoch}_ari_{joint_z_ari:.4f}.csv"
        ),
    )

    # 2. UMAP 降维 (使用 min_dist 和 n_neighbors 优化单细胞展示)
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",  # normalized z 下欧氏距离等价于余弦距离
        random_state=42
    )
    z_umap = reducer.fit_transform(z_joint)

    # ----------------------------------------------------
    # 图 1：按 模态 (Modality) 着色
    # ----------------------------------------------------
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=z_umap[:, 0], y=z_umap[:, 1],
        hue=modality,
        palette={"RNA": "#1f77b4", "ATAC": "#ff7f0e"},
        s=8, linewidth=0, alpha=0.7
    )
    plt.title(f"Best Model Joint Latent Z (by Modality)\nEpoch {epoch} | Joint Z ARI: {joint_z_ari:.4f}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Modality", frameon=False, markerscale=2.5)
    plt.tight_layout()

    path_mod = os.path.join(save_dir, f"best_umap_modality_epoch{epoch}_ari{joint_z_ari:.4f}.png")
    plt.savefig(path_mod, dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 图 2：按 细胞类型 (Cell Type) 着色
    # ----------------------------------------------------
    cell_type_str = cell_type.astype(str)
    unique_types = np.unique(cell_type_str)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=z_umap[:, 0], y=z_umap[:, 1],
        hue=cell_type_str,
        palette="tab20" if len(unique_types) <= 20 else "husl",
        s=8, linewidth=0, alpha=0.7
    )
    plt.title(f"Best Model Joint Latent Z (by Cell Type)\nEpoch {epoch} | Joint Z ARI: {joint_z_ari:.4f}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(
        title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left",
        frameon=False, markerscale=2.5
    )
    plt.tight_layout()

    path_ct = os.path.join(save_dir, f"best_umap_celltype_epoch{epoch}_ari{joint_z_ari:.4f}.png")
    plt.savefig(path_ct, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ UMAP 图像已保存至:\n  1. {path_mod}\n  2. {path_ct}\n")


def gaussian_kernel(x, y, sigma=1.0):

    xx = torch.sum(x*x, dim=1, keepdim=True)
    yy = torch.sum(y*y, dim=1, keepdim=True)

    dist = xx + yy.T - 2*x@y.T

    return torch.exp(
        -dist/(2*sigma*sigma)
    )


def compute_mmd(x,y,sigma=1.0):

    Kxx = gaussian_kernel(x,x,sigma)
    Kyy = gaussian_kernel(y,y,sigma)
    Kxy = gaussian_kernel(x,y,sigma)
    mmd = ( Kxx.mean() + Kyy.mean()-2*Kxy.mean())

    return mmd
def move_batch_to_device(batch_dict, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch_dict.items()}

def variance_loss(z, gamma=1.0):
    std = torch.sqrt(z.var(dim=0)+1e-4)
    return torch.mean(F.relu(gamma-std))

from sklearn.neighbors import NearestNeighbors
import numpy as np

def cross_modality_neighbor_fraction(X, modality_labels, k=30):
    k = min(k + 1, len(X))
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)

    # 删除自己
    idx = idx[:, 1:]

    return np.mean(
        modality_labels[idx] != modality_labels[:, None]
    )


def extract_unpaired_latents_and_labels(model, loader, device):
    """
    针对带 Projector Head 的 UnpairedClusterOTAE 模型的特征提取函数
    """
    model.eval()

    all_z_rna, all_z_atac = [], []
    all_h_rna, all_h_atac = [], []
    all_q_rna, all_q_atac = [], []
    all_labels_rna, all_labels_atac = [], []

    with torch.no_grad():
        for batch_dict in tqdm(loader, desc="Extracting Latents & Projections", leave=False):
            batch_dict = move_batch_to_device(batch_dict, device)
            outputs = model(batch_dict)

            # 1. 提取原始 Latent z (128维)
            all_z_rna.append(outputs["z_rna"].detach().cpu().numpy())
            all_z_atac.append(outputs["z_atac"].detach().cpu().numpy())

            # 2. 🌟 提取 Projector 投影特征 h (64维 L2-Norm)
            if "h_rna" in outputs:
                all_h_rna.append(outputs["h_rna"].detach().cpu().numpy())
                all_h_atac.append(outputs["h_atac"].detach().cpu().numpy())

            # 3. 提取 DEC 软分配概率 q
            all_q_rna.append(outputs["q_rna"].detach().cpu().numpy())
            all_q_atac.append(outputs["q_atac"].detach().cpu().numpy())

            # 4. 收集标签 (分别兼容 RNA 和 ATAC 的独立标签)
            if "label_rna" in batch_dict and batch_dict["label_rna"] is not None:
                all_labels_rna.append(batch_dict["label_rna"].detach().cpu().numpy())
            elif "label" in batch_dict and batch_dict["label"] is not None:
                all_labels_rna.append(batch_dict["label"].detach().cpu().numpy())

            if "label_atac" in batch_dict and batch_dict["label_atac"] is not None:
                all_labels_atac.append(batch_dict["label_atac"].detach().cpu().numpy())
            elif "label" in batch_dict and batch_dict["label"] is not None:
                all_labels_atac.append(batch_dict["label"].detach().cpu().numpy())

    # 拼接矩阵
    latents_dict = {
        "z_rna": np.concatenate(all_z_rna, axis=0),
        "z_atac": np.concatenate(all_z_atac, axis=0),
        "q_rna": np.concatenate(all_q_rna, axis=0),
        "q_atac": np.concatenate(all_q_atac, axis=0)
    }

    if len(all_h_rna) > 0:
        latents_dict["h_rna"] = np.concatenate(all_h_rna, axis=0)
        latents_dict["h_atac"] = np.concatenate(all_h_atac, axis=0)

    y_true_rna = np.concatenate(all_labels_rna, axis=0) if len(all_labels_rna) > 0 else None
    y_true_atac = np.concatenate(all_labels_atac, axis=0) if len(all_labels_atac) > 0 else None

    # 如果是 1:1 Paired 数据且标签一致，返回单个 y_true，否则返回字典
    if y_true_rna is not None and y_true_atac is not None and len(y_true_rna) == len(y_true_atac) and np.array_equal(
            y_true_rna, y_true_atac):
        y_true = y_true_rna
    else:
        y_true = {"rna": y_true_rna, "atac": y_true_atac}

    return latents_dict, y_true


# ================= ================= =================
# 1. 辅助指标计算函数 (纯 NumPy/SciPy/Sklearn 极速无依赖实现)
# ================= ================= =================
def compute_ilisi(X, modality_labels, k=30):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=1).fit(X)
    indices = nn.kneighbors(X, return_distance=False)[:, 1:]  # [N, k]

    lisi_scores = []
    for i in range(len(X)):
        neighbor_modalities = modality_labels[indices[i]]
        # 计算 neighborhood 中各模态的概率分布
        _, counts = np.unique(neighbor_modalities, return_counts=True)
        probs = counts / k
        # Simpson 指数 D = sum(p^2), Inverse Simpson = 1 / D
        simpson_index = np.sum(probs ** 2)
        lisi = 1.0 / (simpson_index + 1e-12)
        lisi_scores.append(lisi)

    return float(np.mean(lisi_scores))


def compute_graph_connectivity(X, cell_types, k=15):
    # 构建全图 kNN 连通图 (无向图对称化)
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=True, n_jobs=-1)
    knn_graph = 0.5 * (knn_graph + knn_graph.T)  # 对称化矩阵

    unique_types = np.unique(cell_types)
    gc_scores = []

    for ct in unique_types:
        mask = (cell_types == ct)
        n_cells = np.sum(mask)

        # 🔍 优化 2: 若该类细胞只有 1 个，其连通性理论上为 1.0，直接存入而不是跳过
        if n_cells <= 1:
            gc_scores.append(1.0)
            continue

        # 提取当前细胞类型的子图
        sub_graph = knn_graph[mask][:, mask]

        # 计算子图的连通分量
        n_components, labels = connected_components(csgraph=sub_graph, directed=False)

        # 找到最大连通分量的细胞数量
        _, counts = np.unique(labels, return_counts=True)
        max_comp_size = np.max(counts)

        # 归一化得分: 最大连通分量占比
        score = max_comp_size / n_cells
        gc_scores.append(score)

    return float(np.mean(gc_scores)) if len(gc_scores) > 0 else 0.0


# ================= ================= =================
# 2. 全面跨模态评估主函数
# ================= ================= =================
def evaluate_unpaired_cluster_performance(model, loader, device, num_clusters=19, random_state=42):
    """
    针对 Unpaired 数据的 Projector 架构全面评估函数
    包含：DEC 聚类指标、KMeans ARI、iLISI 模态混合度、Graph Connectivity 保留度
    """
    latents_dict, y_true = extract_unpaired_latents_and_labels(model, loader, device)

    # 处理标签解包
    if isinstance(y_true, dict):
        y_rna, y_atac = y_true["rna"], y_true["atac"]
    else:
        y_rna, y_atac = y_true, y_true

    if y_rna is None or y_atac is None:
        print("⚠️ [Warning] 未接入真实标签，跳过定量指标评估。")
        return {}

    n_rna = len(y_rna)
    n_atac = len(y_atac)

    # 构建模态指示标签 (0 为 RNA, 1 为 ATAC)
    modality_labels = np.concatenate([np.zeros(n_rna, dtype=int), np.ones(n_atac, dtype=int)], axis=0)

    # ====================================================
    # 1. 单模态评估：RNA 模态
    # ====================================================
    pred_dec_rna = np.argmax(latents_dict["q_rna"], axis=1)
    ari_dec_rna = adjusted_rand_score(y_rna, pred_dec_rna)

    nmi_dec_rna = normalized_mutual_info_score(y_rna, pred_dec_rna)
    pred_km_z_rna = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10).fit_predict(latents_dict["z_rna"])
    ari_km_z_rna = adjusted_rand_score(y_rna, pred_km_z_rna)
    nmi_km_z_rna = normalized_mutual_info_score(y_rna, pred_km_z_rna)
    # ====================================================
    # 2. 单模态评估：ATAC 模态
    # ====================================================
    pred_dec_atac = np.argmax(latents_dict["q_atac"], axis=1)
    ari_dec_atac = adjusted_rand_score(y_atac, pred_dec_atac)

    nmi_dec_atac = normalized_mutual_info_score(y_atac, pred_dec_atac)
    pred_km_z_atac = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10).fit_predict(latents_dict["z_atac"])

    ari_km_z_atac = adjusted_rand_score(y_atac, pred_km_z_atac)
    nmi_km_z_atac = normalized_mutual_info_score(y_atac, pred_km_z_atac)


    # ====================================================
    # 3. 跨模态联合空间评估 (Unpaired 拼接 axis=0)
    # ====================================================
    y_joint = np.concatenate([y_rna, y_atac], axis=0)

    # A. 联合 DEC 聚类评估
    pred_dec_joint = np.concatenate([pred_dec_rna, pred_dec_atac], axis=0)
    ari_dec_joint = adjusted_rand_score(y_joint, pred_dec_joint)
    nmi_dec_joint = normalized_mutual_info_score(y_joint, pred_dec_joint)

    # B. 联合 Latent z 空间 (如 128维) 聚类与对齐评估
    z_joint = np.concatenate([latents_dict["z_rna"], latents_dict["z_atac"]], axis=0)
    pred_km_z_joint = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10).fit_predict(z_joint)
    ari_km_z_joint = adjusted_rand_score(y_joint, pred_km_z_joint)
    nmi_km_z_joint = normalized_mutual_info_score(y_joint, pred_km_z_joint)

    # z 空间的模态混合与图连通性
    ilisi_z = compute_ilisi(z_joint, modality_labels, k=30)
    gc_z = compute_graph_connectivity(z_joint, y_joint, k=30)

    cross_frac = cross_modality_neighbor_fraction(
        z_joint,
        modality_labels,
        k=30
    )
    print(f"Cross-modality neighbor fraction: {cross_frac:.4f}")



    # ====================================================
    # 4. 结果格式化打印
    # ====================================================
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)

    print("\n📊 ==================== [ Unpaired 跨模态聚类与对齐评估报告 ] ====================")
    print(
        f"🔹 [RNA  模态] DEC ARI: {ari_dec_rna:.4f} (NMI: {nmi_dec_rna:.4f}) | Latent z ARI: {ari_km_z_rna:.4f} (NMI: {nmi_km_z_rna:.4f})")
    print(
        f"🔸 [ATAC 模态] DEC ARI: {ari_dec_atac:.4f} (NMI: {nmi_dec_atac:.4f}) | Latent z ARI: {ari_km_z_atac:.4f}(NMI: {nmi_km_z_atac:.4f})")
    print("----------------------------------------------------------------------------------------------------")
    print(f"🌐 [跨模态联合预测] DEC 头混合 ARI: {ari_dec_joint:.4f} (NMI: {nmi_dec_joint:.4f})")
    print(f"                   Latent z 联合空间 KMeans ARI: {ari_km_z_joint:.4f} (NMI: {nmi_km_z_joint:.4f})")
    print("----------------------------------------------------------------------------------------------------")
    print(
        f"🔀 [模态对齐质量 - Latent z]   iLISI (模态混合度, 1~2): {fmt(ilisi_z)} | Graph Connectivity (结构保留, 0~1): {fmt(gc_z)}")
    print("====================================================================================================\n")

    return {
        "rna_dec_ari": ari_dec_rna,
        "atac_dec_ari": ari_dec_atac,
        "joint_dec_ari": ari_dec_joint,
        "joint_z_ari": ari_km_z_joint,
        "ilisi_z": ilisi_z,
        "graph_connectivity_z": gc_z,
    }



class UnifiedModalityEncoder(nn.Module):
    def __init__(self, encoder_config: Dict[str, Any], latent_dim: int = 128):
        super().__init__()
        self.backbone = MLP(encoder_config)

        backbone_out_dim = encoder_config["output"]

        self.shared_head = nn.Sequential(nn.Linear(backbone_out_dim, latent_dim),
                                         nn.LayerNorm(latent_dim),
                                         nn.ReLU())

        nn.init.xavier_normal_(self.shared_head[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z_shared = self.shared_head(h)
        return z_shared


class UnifiedModalityDecoder(nn.Module):
    def __init__(self, decoder_config: Dict[str, Any], latent_dim: int = 128, is_atac: bool = False):
        super().__init__()
        self.is_atac = is_atac

        mlp_in_dim = decoder_config['input']

        if latent_dim != mlp_in_dim:
            self.project_head = nn.Linear(latent_dim, mlp_in_dim)
            nn.init.xavier_normal_(self.project_head.weight)
        else:
            self.project_head = nn.Identity()

        self.backbone = MLP(decoder_config)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project_head(z)
        out = self.backbone(h)

        return out



def compute_partial_paired_cost( z_rna,z_atac, paired_mask, pair_bonus=0):

    z_rna = F.normalize(z_rna,dim=-1)
    z_atac = F.normalize(z_atac,dim=-1)
    cosine = torch.matmul(z_rna,  z_atac.T)
    cost = 1-cosine
    # paired pair 降低固定 margin
    cost = cost - pair_bonus * paired_mask.float()
    cost = cost.clamp(min=0)

    return cost



class HighThroughputPartialOT(nn.Module):
    def __init__(
            self,
            epsilon=0.05,
            transported_mass=0.8,
            dummy_cost_factor=1.5,  # 动态 Dummy Cost 因子 (Mean + factor * Std)
            sinkhorn_iters=50,
            mass_penalty_weight=0.1,
            eps=1e-8
    ):
        super().__init__()
        self.epsilon = epsilon
        self.transported_mass = transported_mass
        self.dummy_cost_factor = dummy_cost_factor
        self.sinkhorn_iters = sinkhorn_iters
        self.mass_penalty_weight = mass_penalty_weight
        self.eps = eps

    def forward(self, C):
        """
        C: 模态间代价矩阵 [N_r, N_a] (例如余弦距离 1 - cos_sim)
        """
        n_r, n_a = C.shape
        device = C.device
        dtype = C.dtype
        m = self.transported_mass

        ################################################
        # 1. 动态计算自适应 Dummy Cost (关键改进)
        ################################################
        with torch.no_grad():
            c_mean = C.mean()
            c_std = C.std()
            # 动态根据当前 Batch 的距离分布设定垃圾桶成本，防止质量逃逸
            dynamic_dummy_cost = (c_mean + self.dummy_cost_factor * c_std).item()

        # 扩展 Cost 矩阵
        C_ext = F.pad(C, (0, 1, 0, 1), value=dynamic_dummy_cost)
        C_ext[-1, -1] = 0.0  # 干净安全地设置 dummy-dummy cost 为 0

        ################################################
        # 2. 构造 Partial Marginals (质量边缘分布)
        ################################################
        real_mass_r = m / n_r
        real_mass_a = m / n_a
        dummy_mass = max(1.0 - m, self.eps)

        a_ext = torch.full((n_r + 1,), real_mass_r, device=device, dtype=dtype)
        a_ext[-1] = dummy_mass

        b_ext = torch.full((n_a + 1,), real_mass_a, device=device, dtype=dtype)
        b_ext[-1] = dummy_mass

        ################################################
        # 3. Log-Domain Sinkhorn 求解器 (No Grad)
        ################################################
        eps = self.epsilon
        with torch.no_grad():
            log_a = torch.log(a_ext + self.eps)
            log_b = torch.log(b_ext + self.eps)
            u = torch.zeros_like(a_ext)
            v = torch.zeros_like(b_ext)
            for _ in range(self.sinkhorn_iters):
                u = eps * (log_a - torch.logsumexp((v[None, :] - C_ext) / eps, dim=1))
                v = eps * (log_b - torch.logsumexp((u[:, None] - C_ext) / eps, dim=0))

            log_pi = (u[:, None] + v[None, :] - C_ext) / eps
            log_pi = torch.clamp(log_pi, min=-80.0, max=80.0)  # 防止 exp 数值下溢/上溢
            pi_ext = torch.exp(log_pi)

            # 截取真实细胞间（非 Dustbin）的传输矩阵 [N_r, N_a]
            pi = pi_ext[:-1, :-1]

        ################################################
        # 4. Envelope Gradient 与 损失目标组装
        ################################################
        # 严格分离求导图：将 pi 视为常数矩阵 (Envelope Theorem)
        pi = pi.detach()

        transport_mass = pi.sum()
        transport_cost = (pi * C).sum()

        # A. 核心匹配代价：平均单细胞对齐距离
        avg_matched_cost = transport_cost / (transport_mass + self.eps)
        # B. 质量约束惩罚项 (Mass Penalty)
        mass_penalty = torch.abs(transport_mass - self.transported_mass)
        # C. 复合损失函数
        ot_loss = avg_matched_cost + self.mass_penalty_weight * mass_penalty

        return {
            "pi": pi,
            "ot_loss": ot_loss,
            "avg_matched_cost": avg_matched_cost,
            "transport_mass": transport_mass,
            "transport_cost": transport_cost,
            "mass_penalty": mass_penalty,
            "dynamic_dummy_cost": dynamic_dummy_cost
        }


#


class UnpairedClusterOTAE(nn.Module):
    def __init__(self, master_config, num_clusters=19, alpha=1.0):
        super().__init__()
        self.config = master_config
        self.latent_dim = master_config.get("latent_dim", 128)
        self.proj_dim = master_config.get("proj_dim", 64)  # 投影后维度，默认 64
        self.num_clusters = num_clusters
        self.alpha = alpha

        # 1. 独立 Encoders
        self.encoder_rna = UnifiedModalityEncoder(master_config["encoder_specs"]["rna"], self.latent_dim)
        self.encoder_atac = UnifiedModalityEncoder(master_config["encoder_specs"]["atac"], self.latent_dim)

        # 2. 独立 Decoders
        self.decoder_rna = UnifiedModalityDecoder(master_config["decoder_specs"]["rna"], self.latent_dim)
        self.decoder_atac = UnifiedModalityDecoder(master_config["decoder_specs"]["atac"], self.latent_dim, is_atac=True)

        # 3. 新增：独立 Projector Heads (用于将 z 映射到对齐空间 h)

        # 4. 模态独立 DEC 聚类中心 (设置在 Projector 空间/或原 Latent 空间)
        self.cluster_layer_rna = nn.Parameter(torch.empty(num_clusters, self.proj_dim))
        self.cluster_layer_atac = nn.Parameter(torch.empty(num_clusters, self.proj_dim))

        nn.init.xavier_uniform_(self.cluster_layer_rna)
        nn.init.xavier_uniform_(self.cluster_layer_atac)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_q(self, h, cluster_layer):
        dist = torch.sum(
            (h.unsqueeze(1) - cluster_layer.unsqueeze(0)) ** 2,
            dim=2
        ).clamp(min=1e-8)
        q = 1 / (1 + dist / self.alpha)
        q = q ** ((self.alpha + 1) / 2)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        return q.clamp(min=1e-8, max=1)


    def forward(self, batch_dict):
        X_RNA = batch_dict["rna"]  # 外部保证 device
        X_ATAC = batch_dict["atac"]
        assert X_RNA.device == self.device

        z_rna = self.encoder_rna(X_RNA)
        z_atac = self.encoder_atac(X_ATAC)

        q_rna = self.compute_q(z_rna, self.cluster_layer_rna)
        q_atac = self.compute_q(z_atac,self.cluster_layer_atac)

        rec_rna = self.decoder_rna(z_rna)
        rec_atac = self.decoder_atac(z_atac)

        tran_rna = self.decoder_rna(z_atac)
        tran_atac = self.decoder_atac(z_rna)


        return {
            "z_rna": z_rna, "z_atac": z_atac,
            "q_rna": q_rna, "q_atac": q_atac,
            "rec_rna": rec_rna, "rec_atac": rec_atac,
            'tran_rna': tran_rna, "tran_atac": tran_atac,
        }

    def compute_dec_loss(self, q):
        eps = 1e-8
        q = q.clamp(min=eps, max=1.0)

        # 算频率 f，加 eps 保护
        freq = q.sum(dim=0, keepdim=True).clamp(min=eps)

        # 计算目标分布 P
        weight = (q ** 2) / freq
        p = weight / (weight.sum(dim=1, keepdim=True).clamp(min=eps))
        p = p.detach().clamp(min=eps, max=1.0)  # 严格阻断 P 的梯度，并 Clamp

        # 稳定计算 KL 散度
        loss = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1).mean()
        return loss



def train():
    # ==========================================
    # 1. 基础环境与配置设置
    # ==========================================
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    batch_size = 256
    pretrain_epochs = 200
    finetune_epochs = 100

    folder_path = r'D:/PycharmProjects/C1/dataset/inhouse/inhouseall_cells_1/'
    # folder_path = r'D:\PycharmProjects\C1\dataset\pbmc10k\PBMCall_cells_1/'

    target_label_col = "cell_type"
    num_clusters = 7

    save_dir = "./checkpoints_paired/inhouse"
    os.makedirs(save_dir, exist_ok=True)

    pretrain_model_path = os.path.join(save_dir, "pretrained_model.pt")
    best_model_path = os.path.join(save_dir, "best_finetuned_model.pt")

    # Loss 权重配置
    pre_lambda_rec_rna = 1.0
    pre_lambda_rec_atac = 1.0
    pre_lambda_var = 1.0
    pre_lambda_cluster_ot = 1.0

    ft_lambda_rec_rna = 1.0
    ft_lambda_rec_atac = 1.0
    ft_lambda_dec = 1.0  # 单模态 DEC Loss 权重
    ft_lambda_cluster_ot = 1  # 跨模态 Cluster-level OT Alignment 权重
    ft_lambda_var = 1.0

    grad_clip_norm = 5.0

    partial_ot = HighThroughputPartialOT(
        epsilon=0.02,
        transported_mass=0.8,
        dummy_cost_factor=1.5,
        sinkhorn_iters=50,
        mass_penalty_weight=0.05
    )


    master_config = {
        "modality_names": ["rna", "atac"],
        "latent_dim": 64,
        'proj_dim': 64,
        "encoder_specs": {
            "rna": {
                "input": 10,
                "hidden": [512],
                "output": 64,
                "use_biases": [True, True],
                "dropouts": [0.1, 0.0],
                "use_batch_norms": [False, False],
                "use_layer_norms": [True, True],
                "activations": ["relu", ""]
            },
            "atac": {
                "input": 2500,
                "hidden": [512],
                "output": 64,
                "use_biases": [True, True],
                "dropouts": [0.1, 0.0],
                "use_batch_norms": [False, False],
                "use_layer_norms": [True, True],
                "activations": ["relu", ""]
            }
        },
        "decoder_specs": {
            "rna": {
                "input": 64,
                "hidden": [512],
                "output": 10,
                "use_biases": [True, True],
                "dropouts": [0.0, 0.0],
                "use_batch_norms": [False, False],
                "use_layer_norms": [True, True],
                "activations": ["relu", ""]
            },
            "atac": {
                "input": 64,
                "hidden": [512],
                "output": 2500,
                "use_biases": [True, True],
                "dropouts": [0.0, 0.0],
                "use_batch_norms": [False, False],
                "use_layer_norms": [True, False],
                "activations": ["relu", ""]
            }
        },
    }

    # ==========================================
    # 2. 数据载入 (天然支持 Unpaired / Paired)
    # ==========================================
    print("⏳ [Data Loading] 正在接入全量数据总线...")
    pipeline_out = load_real_folder_pipeline(
        folder_path=folder_path,
        mode="paired",
        batch_size=batch_size,
        num_workers=0,
        label_col=target_label_col,
        n_top_genes=2500,
        n_top_peaks=2500
    )
    train_loader = pipeline_out["dataloader"]
    dataset = pipeline_out["dataset"]

    print(f"✅ DataLoader 构建完成，batch 数量: {len(train_loader)}")

    # ==========================================
    # 3. 初始化新网络架构与优化器
    model = UnpairedClusterOTAE(
        master_config,
        num_clusters=num_clusters
    ).to(device)


    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    # ==========================================
    # STAGE 1: 自编码骨干网独立预训练 (仅重构 + 方差正则，不推扯空间)
    # ==========================================
    print("\n📦 === [STAGE 1] 开启自编码骨干网预训练: Reconstruction + Variance Regularization ===")

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "rec_rna": 0.0, "rec_atac": 0.0, 'var': 0.0, 'ot': 0.0,
                        'tran_rna': 0.0, 'tran_atac': 0.0}

        for batch_dict in tqdm(train_loader, desc=f"Pretrain Epoch {epoch}", leave=False):
            batch_dict = move_batch_to_device(batch_dict, device)
            rna_idx = batch_dict["rna_idx"]
            atac_idx = batch_dict["atac_idx"]
            batch_pair_mask = dataset.get_batch_pair_mask(rna_idx, atac_idx).cuda()
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_dict)


            # 1. 独立重构损失
            loss_rec_rna = F.mse_loss(outputs["rec_rna"], batch_dict["rna"])
            loss_rec_atac = F.mse_loss(outputs["rec_atac"], batch_dict["atac"])

            loss_tran_rna = F.mse_loss(outputs["tran_rna"], batch_dict["rna"])
            loss_tran_atac = F.mse_loss(outputs["tran_atac"], batch_dict["atac"])

            # 2. 防止表示坍缩的方差损失
            loss_var = variance_loss(outputs["z_rna"]) + variance_loss(outputs["z_atac"])

            Cost = compute_partial_paired_cost(outputs["z_rna"],outputs["z_atac"],batch_pair_mask)
            ot_result = partial_ot(Cost)
            loss_cluster_ot = ot_result["ot_loss"]


            loss_total = (
                    pre_lambda_rec_rna * loss_rec_rna
                    + pre_lambda_rec_atac * loss_rec_atac
                    + pre_lambda_var * loss_var
                    + pre_lambda_cluster_ot * loss_cluster_ot
                    + loss_tran_rna + loss_tran_atac
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_losses["total"] += loss_total.item()
            epoch_losses["rec_rna"] += loss_rec_rna.item()
            epoch_losses["rec_atac"] += loss_rec_atac.item()
            epoch_losses['tran_rna'] += loss_tran_rna.item()
            epoch_losses['tran_atac'] += loss_tran_atac.item()
            epoch_losses['var'] += loss_var.item()
            epoch_losses['ot'] += loss_cluster_ot.item()

        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        if epoch % 10 == 0 or epoch == pretrain_epochs:
            print(
                f"📉 Pretrain Epoch [{epoch}/{pretrain_epochs}] "
                f"| Total: {avg_losses['total']:.6f} "
                f"| RNA Rec: {avg_losses['rec_rna']:.6f} "
                f"| ATAC Rec: {avg_losses['rec_atac']:.6f} "
                f"| Tran RNA: {avg_losses['tran_rna']:.6f} "
                f"| Tran ATAC: {avg_losses['tran_atac']:.6f}"
                f"| Var: {avg_losses['var']:.6f}"
                f"| OT: {avg_losses['ot']:.6f} "
            )

    torch.save(model.state_dict(), pretrain_model_path)
    print(f"💾 预训练模型已保存: {pretrain_model_path}")

    # ==========================================
    # STAGE 2: 分别抽取各自模态的隐特征，完成独立 KMeans 初始化
    # ==========================================
    print("\n🔮 === [STAGE 2] 提取独立隐特征并对 RNA / ATAC 分别 KMeans 初始化聚类中心 ===")

    # 抽取各自模态隐特征的辅助函数 (返回字典: {'z_rna': tensor, 'z_atac': tensor})
    all_latents, all_true_labels = extract_unpaired_latents_and_labels(model=model, loader=train_loader, device=device)

    # 1. RNA KMeans 初始化
    kmeans_rna = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    kmeans_rna.fit(all_latents['z_rna'])
    with torch.no_grad():
        model.cluster_layer_rna.copy_(torch.tensor(kmeans_rna.cluster_centers_, dtype=torch.float32, device=device))

    # 2. ATAC KMeans 初始化
    kmeans_atac = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
    kmeans_atac.fit(all_latents['z_atac'])
    with torch.no_grad():
        model.cluster_layer_atac.copy_(torch.tensor(kmeans_atac.cluster_centers_, dtype=torch.float32, device=device))

    if all_true_labels is not None:
        ari_rna = adjusted_rand_score(all_true_labels, kmeans_rna.labels_)
        nmi_rna = normalized_mutual_info_score(all_true_labels, kmeans_rna.labels_)
        ari_atac = adjusted_rand_score(all_true_labels, kmeans_atac.labels_)
        nmi_atac = normalized_mutual_info_score(all_true_labels, kmeans_atac.labels_)
        print(f"📈 [Stage 2 RNA KMeans Baseline]  ARI: {ari_rna:.4f} | NMI: {nmi_rna:.4f}")
        print(f"📈 [Stage 2 ATAC KMeans Baseline] ARI: {ari_atac:.4f} | NMI: {nmi_atac:.4f}")

    # ==========================================
    # STAGE 3: 单模态 DEC + Cluster-level OT Alignment 联合微调
    # ==========================================
    print("\n🔥 === [STAGE 3] 开启 Reconstruction + Modality-DEC + Cluster-OT 联合微调 ===")
    best_joint_z_ari = -1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-4

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0, "rec_rna": 0.0, "rec_atac": 0.0,
            "dec_rna": 0.0, "dec_atac": 0.0, "cluster_ot": 0.0, "var": 0.0
        }
        epoch_diag = {"top3": 0., "top5": 0.,"mmd": 0., "cost": 0.,"sim": 0.,"entropy": 0.,"support": 0.,"max_prob": 0.,}

        for batch_dict in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}", leave=False):
            batch_dict = move_batch_to_device(batch_dict, device)
            rna_idx = batch_dict["rna_idx"]
            atac_idx = batch_dict["atac_idx"]
            batch_pair_mask = dataset.get_batch_pair_mask(rna_idx, atac_idx).cuda()
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_dict)

            # 1. 重构损失
            loss_rec_rna = F.mse_loss(outputs["rec_rna"], batch_dict["rna"])
            loss_rec_atac = F.mse_loss(outputs["rec_atac"], batch_dict["atac"])

            # 2. 模态独立 DEC Loss (挤压与分离各模态内部 Cluster)
            loss_dec_rna = model.compute_dec_loss(outputs["q_rna"])
            loss_dec_atac = model.compute_dec_loss(outputs["q_atac"])

            # 3. 跨模态语义对齐: 基于概率分布的 Cluster-level OT
            Cost = compute_partial_paired_cost(outputs["z_rna"],
                                               outputs["z_atac"],
                                               batch_pair_mask)
            ot_result = partial_ot(Cost)
            loss_cluster_ot = ot_result["ot_loss"]
            pi = ot_result["pi"]

            with torch.no_grad():
                B = pi.shape[0]

                # OT entropy
                pi_row = pi / (pi.sum(1, keepdim=True) + 1e-8)
                entropy = (-(pi_row * torch.log(pi_row + 1e-8)).sum(1).mean())
                support = torch.exp(entropy)
                max_prob = (pi_row.max(1).values.mean())
                top3 = (torch.topk(pi_row,min(3, B),dim=1).values.sum(1).mean())
                top5 = (torch.topk(pi_row,min(5, B),dim=1).values.sum(1).mean())

                # cost
                mean_cost = Cost.mean()

                # similarity
                sim = 1 - Cost
                mean_sim = sim.mean()

                # MMD
                mmd = compute_mmd(outputs["z_rna"],outputs["z_atac"])

            # 4. 方差正则
            loss_var = variance_loss(outputs["z_rna"]) + variance_loss(outputs["z_atac"])
            # 总体联合损失
            total_loss = (
                    ft_lambda_rec_rna * loss_rec_rna
                    + ft_lambda_rec_atac * loss_rec_atac
                    + ft_lambda_dec * (loss_dec_rna + loss_dec_atac)
                    + ft_lambda_cluster_ot * loss_cluster_ot
                    + ft_lambda_var * loss_var
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            # 累加监控指标
            epoch_diag["entropy"] += entropy.item()
            epoch_diag["support"] += support.item()
            epoch_diag["max_prob"] += max_prob.item()
            epoch_diag["top3"] += top3.item()
            epoch_diag["top5"] += top5.item()
            epoch_diag["cost"] += mean_cost.item()
            epoch_diag["sim"] += mean_sim.item()
            epoch_diag["mmd"] += mmd.item()

            # 记录累加损失
            epoch_losses["total"] += total_loss.item()
            epoch_losses["rec_rna"] += loss_rec_rna.item()
            epoch_losses["rec_atac"] += loss_rec_atac.item()
            epoch_losses["dec_rna"] += loss_dec_rna.item()
            epoch_losses["dec_atac"] += loss_dec_atac.item()
            epoch_losses["cluster_ot"] += loss_cluster_ot.item()
            epoch_losses["var"] += loss_var.item()

        n_batches = len(train_loader)
        avg_loss = {k: v / n_batches for k, v in epoch_losses.items()}

        print(
            f"📉 Fine-tune Epoch [{epoch}/{finetune_epochs}] "
            f"| Total: {avg_loss['total']:.6f} "
            f"| RNA Rec: {avg_loss['rec_rna']:.4f} "
            f"| ATAC Rec: {avg_loss['rec_atac']:.4f} "
            f"| DEC RNA: {avg_loss['dec_rna']:.4f} "
            f"| DEC ATAC: {avg_loss['dec_atac']:.4f} "
            f"| Cluster OT: {avg_loss['cluster_ot']:.4f}"
        )
        print(
            f"Transport: {ot_result['transport_cost'].item():.6f} "
            f"  Entropy: {epoch_diag['entropy'] / n_batches:.3f}"
            f"  Support: {epoch_diag['support'] / n_batches:.2f}"
            f"  MaxProb: {epoch_diag['max_prob'] / n_batches:.4f}"
            f"  Top3 Mass: {epoch_diag['top3'] / n_batches:.3f}"
            f"  Top5 Mass: {epoch_diag['top5'] / n_batches:.3f}"
            f"  Cost: {epoch_diag['cost'] / n_batches:.4f}"
            f"  Similarity: {epoch_diag['sim'] / n_batches:.4f}"
            f"  MMD: {epoch_diag['mmd'] / n_batches:.6f}"
        )
        print("-" * 90)
        # 每个 epoch 结束后，用稳定的 eval 模式全量评估 DEC 聚类结果
        matric = evaluate_unpaired_cluster_performance(
            model=model,
            loader=train_loader,
            device=device,
            num_clusters=num_clusters,
            random_state=42
        )
        current_joint_z_ari = matric["joint_z_ari"]


        # 3. 如果 joint_z_ari 刷新最佳历史纪录，保存模型并画 UMAP
        if current_joint_z_ari > best_joint_z_ari:
            best_joint_z_ari = current_joint_z_ari
            best_model_path = os.path.join(save_dir, f"best_joint_z_ari_model{current_joint_z_ari}.pt")

            # 保存 PyTorch 权重
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 刷新最佳纪录! Joint Z ARI={best_joint_z_ari:.4f}，模型已保存。")

            # 绘制 UMAP
            plot_best_joint_z_umap(
                model=model,
                loader=train_loader,
                device=device,
                save_dir=save_dir,
                epoch=epoch,
                joint_z_ari=best_joint_z_ari
            )
    # torch.save(model.state_dict(), final_model_path)
    # print(f"\n💾 最终模型已保存: {final_model_path}")
    # if best_ari >= 0:
    #     print(f"[Eval KMeans Model: ARI={best_ari:.4f} | NMI={best_nmi:.4f} | Path={best_model_path}")

    return model




if __name__ == "__main__":
    trained_model = train()





