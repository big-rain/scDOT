


from Models.MLP import MLP

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
from dataloader.dataset_MET import build_patchseq_data

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



def gaussian_kernel(x, y, sigma=1.0):

    xx = torch.sum(x*x, dim=1, keepdim=True)
    yy = torch.sum(y*y, dim=1, keepdim=True)

    dist = xx + yy.T - 2*x@y.T

    return torch.exp(
        -dist/(2*sigma*sigma)
    )



def move_batch_to_device(batch_dict, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch_dict.items()}

def variance_loss(z, gamma=1.0):
    std = torch.sqrt(z.var(dim=0)+1e-4)
    return torch.mean(F.relu(gamma-std))

from sklearn.neighbors import NearestNeighbors
import numpy as np







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


import torch
import torch.nn as nn

import torch
import torch.nn as nn


class UnpairedClusterOTAE3Mod(nn.Module):
    def __init__(self, master_config, num_clusters=15, alpha=1.0):
        super().__init__()
        self.config = master_config
        self.latent_dim = master_config.get("latent_dim", 128)
        self.proj_dim = master_config.get("proj_dim", 64)
        self.num_clusters = num_clusters
        self.alpha = alpha

        # 1. 三模态独立 Encoders (RNA, Ephys, Morph)
        self.encoder_rna = UnifiedModalityEncoder(master_config["encoder_specs"]["rna"], self.latent_dim)
        self.encoder_ephys = UnifiedModalityEncoder(master_config["encoder_specs"]["ephys"], self.latent_dim)
        self.encoder_morph = UnifiedModalityEncoder(master_config["encoder_specs"]["morph"], self.latent_dim)

        # 2. 三模态独立 Decoders
        self.decoder_rna = UnifiedModalityDecoder(master_config["decoder_specs"]["rna"], self.latent_dim)
        self.decoder_ephys = UnifiedModalityDecoder(master_config["decoder_specs"]["ephys"], self.latent_dim)
        self.decoder_morph = UnifiedModalityDecoder(master_config["decoder_specs"]["morph"], self.latent_dim)

        # 3. 三模态各自的聚类中心 (每个模态独立学习对齐空间中的 15 个生物亚型中心)
        self.cluster_layer_rna = nn.Parameter(torch.empty(num_clusters, self.latent_dim))
        self.cluster_layer_ephys = nn.Parameter(torch.empty(num_clusters, self.latent_dim))
        self.cluster_layer_morph = nn.Parameter(torch.empty(num_clusters, self.latent_dim))

        # 参数初始化
        nn.init.xavier_uniform_(self.cluster_layer_rna)
        nn.init.xavier_uniform_(self.cluster_layer_ephys)
        nn.init.xavier_uniform_(self.cluster_layer_morph)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_cluster_assignment(self, z, cluster_layer):
        """
        输入:
            z: [Batch, latent_dim]
            cluster_layer: [num_clusters, latent_dim]
        输出:
            cluster_logits: [Batch, num_clusters] (负欧氏距离平方，专供 SHAP 解释与硬分配判别)
            q: [Batch, num_clusters] (DEC Student-t 软分配概率分布，用于 DEC Loss 优化)
        """
        # dist_sq: [Batch, num_clusters]
        dist_sq = torch.sum(
            (z.unsqueeze(1) - cluster_layer.unsqueeze(0)) ** 2,
            dim=-1
        ).clamp(min=1e-8)

        # 1. Logits 取负距离：距离中心越近，值越大 (SHAP 核心目标)
        cluster_logits = -dist_sq

        # 2. DEC Student-t 软分配概率
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        q = q.clamp(min=1e-8, max=1.0)

        return cluster_logits, q

    def forward(self, batch_dict):
        # 1. 获取输入
        X_RNA = batch_dict["rna"]
        X_EPHYS = batch_dict["ephys"]
        X_MORPH = batch_dict["morph"]

        # 2. 特征编码获得隐空间表征 z
        z_rna = self.encoder_rna(X_RNA)
        z_ephys = self.encoder_ephys(X_EPHYS)
        z_morph = self.encoder_morph(X_MORPH)

        # 3. 计算聚类 Logits (SHAP 解释用) 与 DEC 软分配 q
        logits_rna, q_rna = self.compute_cluster_assignment(z_rna, self.cluster_layer_rna)
        logits_ephys, q_ephys = self.compute_cluster_assignment(z_ephys, self.cluster_layer_ephys)
        logits_morph, q_morph = self.compute_cluster_assignment(z_morph, self.cluster_layer_morph)

        # 4. 本模态自身重构 (Self-Reconstruction)
        rec_rna = self.decoder_rna(z_rna)
        rec_ephys = self.decoder_ephys(z_ephys)
        rec_morph = self.decoder_morph(z_morph)

        # 5. 跨模态生成/翻译 (Cross-Modal Translation)
        # RNA -> Ephys / Morph (阶段二跨模态弦图核心路径)
        tran_rna_to_ephys = self.decoder_ephys(z_rna)
        tran_rna_to_morph = self.decoder_morph(z_rna)

        # Ephys -> RNA / Morph
        tran_ephys_to_rna = self.decoder_rna(z_ephys)
        tran_ephys_to_morph = self.decoder_morph(z_ephys)

        # Morph -> RNA / Ephys
        tran_morph_to_rna = self.decoder_rna(z_morph)
        tran_morph_to_ephys = self.decoder_ephys(z_morph)

        return {
            # 隐空间表征
            "z_rna": z_rna, "z_ephys": z_ephys, "z_morph": z_morph,

            # SHAP 解释专用的 Logits
            "logits_rna": logits_rna, "logits_ephys": logits_ephys, "logits_morph": logits_morph,

            # DEC 训练分布
            "q_rna": q_rna, "q_ephys": q_ephys, "q_morph": q_morph,

            # 自重构
            "rec_rna": rec_rna, "rec_ephys": rec_ephys, "rec_morph": rec_morph,

            # 跨模态翻译输出
            "tran_rna_to_ephys": tran_rna_to_ephys,
            "tran_rna_to_morph": tran_rna_to_morph,
            "tran_ephys_to_rna": tran_ephys_to_rna,
            "tran_ephys_to_morph": tran_ephys_to_morph,
            "tran_morph_to_rna": tran_morph_to_rna,
            "tran_morph_to_ephys": tran_morph_to_ephys,
        }

    def compute_dec_loss(self, q):
        """标准 DEC 目标分布 P 构造及 KL 散度损失"""
        eps = 1e-8
        q = q.clamp(min=eps, max=1.0)
        freq = q.sum(dim=0, keepdim=True).clamp(min=eps)

        weight = (q ** 2) / freq
        p = weight / (weight.sum(dim=1, keepdim=True).clamp(min=eps))
        p = p.detach().clamp(min=eps, max=1.0)

        loss = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1).mean()
        return loss


import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm


def extract_3mod_latent_and_labels(model, loader, device, label_key="fine"):
    """
    辅助函数：提取三模态全量隐特征 z_rna, z_ephys, z_morph
    并将三者拼接融合为 z_joint [num_cells, latent_dim * 3]
    """
    model.eval()
    z_rna_list, z_ephys_list, z_morph_list = [], [], []
    labels_list = []

    with torch.no_grad():
        for batch_dict in loader:
            batch_dict = move_batch_to_device(batch_dict, device)
            outputs = model(batch_dict)

            z_rna_list.append(outputs["z_rna"].cpu().numpy())
            z_ephys_list.append(outputs["z_ephys"].cpu().numpy())
            z_morph_list.append(outputs["z_morph"].cpu().numpy())

            if "labels" in batch_dict and label_key in batch_dict["labels"]:
                labels_list.append(batch_dict["labels"][label_key].cpu().numpy())

    z_rna = np.vstack(z_rna_list)       # 形状: [num_cells, latent_dim]
    z_ephys = np.vstack(z_ephys_list)   # 形状: [num_cells, latent_dim]
    z_morph = np.vstack(z_morph_list)   # 形状: [num_cells, latent_dim]

    # 👉 核心修改：使用 concat 水平拼接三模态表征，保留各模态独特互补信息
    # 形状为: [num_cells, latent_dim * 3] (例如 64 * 3 = 192 维)
    z_joint = np.concatenate([z_rna, z_ephys, z_morph], axis=1)

    labels = np.concatenate(labels_list) if len(labels_list) > 0 else None
    return z_joint, z_rna, z_ephys, z_morph, labels

def compute_prototype_assignment(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    tau: float = 0.1,
):
    z = F.normalize(z, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    logits = torch.matmul(z, prototypes.T) / tau
    assignment = F.softmax(logits, dim=1)

    return assignment


def build_hybrid_cost(
    z_rna: torch.Tensor,
    z_atac: torch.Tensor,
    prototype_rna: torch.Tensor,
    prototype_atac: torch.Tensor,
    lambda_semantic: float = 0.3,
    tau: float = 0.1,
):
    # Geometry Cost
    z_rna = F.normalize(z_rna, dim=1)
    z_atac = F.normalize(z_atac, dim=1)
    geometry_cost = 1.0 - torch.matmul(z_rna,z_atac.T)

    # Prototype Assignment
    assign_rna = compute_prototype_assignment( z_rna,prototype_rna,tau)
    assign_atac = compute_prototype_assignment(z_atac,prototype_atac,tau)

    # Semantic Cost
    semantic_similarity = torch.matmul(assign_rna,assign_atac.T)
    semantic_cost = 1.0 - semantic_similarity

    cost = geometry_cost
    # Hybrid Cost
    cost = geometry_cost + lambda_semantic * semantic_cost

    return cost, {
        "geometry_cost": geometry_cost,
        "semantic_cost": semantic_cost,
        "assign_rna": assign_rna,
        "assign_atac": assign_atac,
    }


def train():
    # ==========================================
    # 1. 基础环境与超参数设置
    # ==========================================
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    batch_size = 512
    pretrain_epochs = 400
    finetune_epochs = 300
    num_clusters = 27
    clusters = 16

    rna_path = "../dataset/PatchseqGABAergicNeuron/adata_RNA_TEM.h5ad"
    ephys_path = "../dataset/PatchseqGABAergicNeuron/adata_Ephys_TEM.h5ad"
    morph_path = "../dataset/PatchseqGABAergicNeuron/adata_Morph_TEM.h5ad"
    fine_label_key = "cell_type_TEM"

    save_dir = "./check/patchseq"
    os.makedirs(save_dir, exist_ok=True)

    pretrain_model_path = os.path.join(save_dir, "patchseq_pretrained_model"
                                                 ""
                                                 ".pt")
    best_model_path = os.path.join(save_dir, "patchseq_best_finetuned_model.pt")
    ot_layer = HighThroughputPartialOT(
        epsilon=0.02,
        transported_mass=0.8,
        dummy_cost_factor=1.5,
        sinkhorn_iters=50,
        mass_penalty_weight=0.05
    )

    # ==========================================
    # 2. 模态权重与损失超参数
    # ==========================================
    pre_lambda_rec = 1.0       # 自重构权重
    pre_lambda_tran = 1     # 跨模态重构/翻译权重
    pre_lambda_ot = 1.0        # 最优传输权重

    ft_lambda_rec = 1.0
    ft_lambda_tran =1
    ft_lambda_ot = 1.0
    ft_lambda_cluster = 1.0    # 三模态 DEC 聚类权重
    grad_clip_norm = 5.0

    # ==========================================
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

    # ==========================================
    # 4. 数据总线接入
    # ==========================================
    print("⏳ [Data Loading] 正在接入全量 Patch-seq 三模态数据总线...")
    cv = 0
    ind_dict_file = "../dataset/PatchseqGABAergicNeuron/cv_5_ind_dict.pkl"
    if os.path.exists(ind_dict_file):
        with open(ind_dict_file, "rb") as f:
            ind_dict = pickle.load(f)
        print("✓ 载入已有 ind_dict.pkl 交叉验证划分")
    else:
        raise FileNotFoundError(f"未找到指定的切分文件: {ind_dict_file}")

    data = build_patchseq_data(
        rna_input=rna_path,
        ephys_input=ephys_path,
        morph_input=morph_path,
        fine_label_key=fine_label_key,
        ind_dict=ind_dict,
        cv=cv,
        batch_size=batch_size,
        split=False,
        device=device,
    )

    all_loader = data["all"]["loader"]

    # 固化元数据
    meta_info = {
        "fine_classes": data["meta"]["fine_classes"],
        "coarse_classes": data["meta"]["coarse_classes"],
        "fine_to_index": data["meta"]["fine_to_index"],
        "index_to_fine": {i: name for i, name in enumerate(data["meta"]["fine_classes"])},
        "fine_to_coarse_idx": data["meta"]["fine_to_coarse_idx"].tolist(),
    }
    with open("patchseq_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)
    print("✅ 元数据已安全固化到 patchseq_metadata.json！")

    # ==========================================
    # 5. 模型与优化器初始化
    # ==========================================
    model = UnpairedClusterOTAE3Mod(
        master_config=master_config,
        num_clusters=num_clusters,
        alpha=1.0
    ).to(device)



    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    # ==========================================
    # STAGE 1: 自编码 + 跨模态翻译 + 最优传输预训练
    # ==========================================
    print("\n📦 === [STAGE 1] 开启三模态自编码骨干网预训练 (All Data) ===")
    pre_best_ari = -1.0
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "rec": 0.0, "tran": 0.0, "ot": 0.0}

        for batch_dict in tqdm(all_loader, desc=f"Pretrain Epoch {epoch}", leave=False):
            batch_dict = move_batch_to_device(batch_dict, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_dict)

            # 1. 自重构损失 (Self-Reconstruction)
            loss_rec = (
                F.mse_loss(outputs["rec_rna"], batch_dict["rna"])
                + F.mse_loss(outputs["rec_ephys"], batch_dict["ephys"])
                + F.mse_loss(outputs["rec_morph"], batch_dict["morph"])
            )

            # 2. 跨模态翻译/生成损失 (Cross Translation Loss)
            loss_tran = (
                F.mse_loss(outputs["tran_rna_to_ephys"], batch_dict["ephys"])
                + F.mse_loss(outputs["tran_rna_to_morph"], batch_dict["morph"])
                + F.mse_loss(outputs["tran_ephys_to_rna"], batch_dict["rna"])
                + F.mse_loss(outputs["tran_ephys_to_morph"], batch_dict["morph"])
                + F.mse_loss(outputs["tran_morph_to_rna"], batch_dict["rna"])
                + F.mse_loss(outputs["tran_morph_to_ephys"], batch_dict["ephys"])
            )

            # 3. 三模态跨模态最优传输 (RNA ↔ Ephys, RNA ↔ Morph)
            cost_re, _ = build_hybrid_cost(
                outputs["z_rna"], outputs["z_ephys"],
                model.cluster_layer_rna, model.cluster_layer_ephys,
                lambda_semantic=0.3,
            )
            cost_rm, _ = build_hybrid_cost(
                outputs["z_rna"], outputs["z_morph"],
                model.cluster_layer_rna, model.cluster_layer_morph,
                lambda_semantic=0.3,
            )

            loss_ot = ot_layer(cost_re)["ot_loss"] + ot_layer(cost_rm)["ot_loss"]

            loss_total = (
                pre_lambda_rec * loss_rec
                # + pre_lambda_tran * loss_tran
                + pre_lambda_ot * loss_ot
            )

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_losses["total"] += loss_total.item()
            epoch_losses["rec"] += loss_rec.item()
            epoch_losses["tran"] += loss_tran.item()
            epoch_losses["ot"] += loss_ot.item()

        n_batches = len(all_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == pretrain_epochs:
            z_joint, _, _, _, cur_labels = extract_3mod_latent_and_labels(model, all_loader, device)
            val_kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
            cur_preds = val_kmeans.fit_predict(z_joint)

            cur_ari = adjusted_rand_score(cur_labels, cur_preds) if cur_labels is not None else 0.0
            cur_nmi = normalized_mutual_info_score(cur_labels, cur_preds) if cur_labels is not None else 0.0

            print(
                f"📉 Pretrain [{epoch:03d}/{pretrain_epochs}] "
                f"| Total: {avg_losses['total']:.4f} (Rec: {avg_losses['rec']:.3f}, Tran: {avg_losses['tran']:.3f}, OT: {avg_losses['ot']:.4f}) "
                f"| KMeans ARI: {cur_ari:.4f} | NMI: {cur_nmi:.4f}"
            )

            if cur_ari > pre_best_ari:
                pre_best_ari = cur_ari
                torch.save(model.state_dict(), pretrain_model_path)
                df_latent = pd.DataFrame(z_joint, columns=[f"z_{i}" for i in range(z_joint.shape[1])])
                df_latent["true_label"] = cur_labels
                df_latent["kmeans_pred"] = cur_preds
                df_latent.to_csv(os.path.join(save_dir, "pre_patchseq_best_latent_z.csv"), index=False)
                print(f"<UNK> Pretrain [{epoch:03d}/{pretrain_epochs}] ")
            if epoch == pretrain_epochs:
                df_latent = pd.DataFrame(z_joint, columns=[f"z_{i}" for i in range(z_joint.shape[1])])
                df_latent["true_label"] = cur_labels
                df_latent["kmeans_pred"] = cur_preds
                df_latent.to_csv(os.path.join(save_dir, "final_patchseq_best_latent_z.csv"), index=False)
                print(f"<UNK> Pretrain [{epoch:03d}/{pretrain_epochs}] ")

    # ==========================================
    # STAGE 2: KMeans 初始化三模态 DEC 聚类中心
    # ==========================================
    # ==========================================
    # STAGE 2: 特征标准化与 KMeans 初始化聚类中心 (提升各模态与联合 ARI/NMI)
    # ==========================================
    print("\n🔮 === [STAGE 2] 提取隐特征并通过优化 KMeans 初始化各模态聚类中心 ===")
    z_joint_raw, z_rna_raw, z_ephys_raw, z_morph_raw, all_true_labels = extract_3mod_latent_and_labels(
        model, all_loader, device
    )

    # 1. 对隐特征进行模态内 L2 归一化（消除模态间尺度方差差异，大幅提升几何聚类紧致度与 ARI/NMI）
    def normalize_feature(x):
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norm

    z_rna_norm = normalize_feature(z_rna_raw)
    z_ephys_norm = normalize_feature(z_ephys_raw)
    z_morph_norm = normalize_feature(z_morph_raw)

    # 归一化后拼接成高质量联合空间
    z_joint_norm = np.concatenate([z_rna_norm, z_ephys_norm, z_morph_norm], axis=1)

    # 2. 在联合空间做高精度 KMeans (多重随机初始化探索全局最优解)
    kmeans_joint = KMeans(n_clusters=num_clusters, n_init=50, max_iter=500, random_state=42)
    kmeans_joint.fit(z_joint_norm)

    # 3. 将高质量联合中心分块赋给各模态，并在原始尺度/单位球面上进行聚类中心校准
    centers_joint = kmeans_joint.cluster_centers_  # [num_clusters, 192]

    # 根据聚类分配计算各模态在原始隐空间的真实质心（Centroids）
    cluster_assigns = kmeans_joint.labels_
    centers_rna = np.zeros((num_clusters, latent_dim), dtype=np.float32)
    centers_ephys = np.zeros((num_clusters, latent_dim), dtype=np.float32)
    centers_morph = np.zeros((num_clusters, latent_dim), dtype=np.float32)

    for k in range(num_clusters):
        mask = (cluster_assigns == k)
        if np.sum(mask) > 0:
            centers_rna[k] = z_rna_raw[mask].mean(axis=0)
            centers_ephys[k] = z_ephys_raw[mask].mean(axis=0)
            centers_morph[k] = z_morph_raw[mask].mean(axis=0)
        else:
            # 孤立簇后备策略
            centers_rna[k] = z_rna_raw.mean(axis=0)
            centers_ephys[k] = z_ephys_raw.mean(axis=0)
            centers_morph[k] = z_morph_raw.mean(axis=0)

    # 拷贝中心参数至模型
    with torch.no_grad():
        model.cluster_layer_rna.copy_(torch.tensor(centers_rna, dtype=torch.float32, device=device))
        model.cluster_layer_ephys.copy_(torch.tensor(centers_ephys, dtype=torch.float32, device=device))
        model.cluster_layer_morph.copy_(torch.tensor(centers_morph, dtype=torch.float32, device=device))

    # 4. 详细评估输出：三模态各自独立指标与联合指标
    if all_true_labels is not None:
        # 单模态各自跑 KMeans 评估判别力
        km_rna = KMeans(n_clusters=num_clusters, n_init=20, random_state=42).fit(z_rna_norm)
        km_ephys = KMeans(n_clusters=num_clusters, n_init=20, random_state=42).fit(z_ephys_norm)
        km_morph = KMeans(n_clusters=num_clusters, n_init=20, random_state=42).fit(z_morph_norm)

        ari_rna = adjusted_rand_score(all_true_labels, km_rna.labels_)
        nmi_rna = normalized_mutual_info_score(all_true_labels, km_rna.labels_)

        ari_ephys = adjusted_rand_score(all_true_labels, km_ephys.labels_)
        nmi_ephys = normalized_mutual_info_score(all_true_labels, km_ephys.labels_)

        ari_morph = adjusted_rand_score(all_true_labels, km_morph.labels_)
        nmi_morph = normalized_mutual_info_score(all_true_labels, km_morph.labels_)

        # 联合空间指标
        ari_joint = adjusted_rand_score(all_true_labels, kmeans_joint.labels_)
        nmi_joint = normalized_mutual_info_score(all_true_labels, kmeans_joint.labels_)

        print("=" * 70)
        print("📊 [Stage 2 各模态表征质量评估 - Baseline Evaluation]")
        print(f"  🧬 RNA Modality   -> ARI: {ari_rna:.4f} | NMI: {nmi_rna:.4f}")
        print(f"  ⚡ Ephys Modality -> ARI: {ari_ephys:.4f} | NMI: {nmi_ephys:.4f}")
        print(f"  🌳 Morph Modality -> ARI: {ari_morph:.4f} | NMI: {nmi_morph:.4f}")
        print(f"  🔗 Joint 3-Modal  -> ARI: {ari_joint:.4f} | NMI: {nmi_joint:.4f}")
        print("=" * 70)

    # ==========================================
    # STAGE 3: 重构 + 跨模态翻译 + 对齐 + DEC 三模态联合微调
    # ==========================================
    print("\n🔥 === [STAGE 3] 开启全量三模态联合微调 (Rec + Tran + OT + DEC) ===")
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-4

    best_ari = -1.0
    best_nmi = -1.0

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "rec": 0.0, "tran": 0.0, "ot": 0.0, "cluster": 0.0}

        for batch_dict in tqdm(all_loader, desc=f"Fine-tune Epoch {epoch}", leave=False):
            batch_dict = move_batch_to_device(batch_dict, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch_dict)

            # 1. 自重构损失
            loss_rec = (
                F.mse_loss(outputs["rec_rna"], batch_dict["rna"])
                + F.mse_loss(outputs["rec_ephys"], batch_dict["ephys"])
                + F.mse_loss(outputs["rec_morph"], batch_dict["morph"])
            )

            # 2. 跨模态重构损失
            loss_tran = (
                F.mse_loss(outputs["tran_rna_to_ephys"], batch_dict["ephys"])
                + F.mse_loss(outputs["tran_rna_to_morph"], batch_dict["morph"])
                + F.mse_loss(outputs["tran_ephys_to_rna"], batch_dict["rna"])
                + F.mse_loss(outputs["tran_ephys_to_morph"], batch_dict["morph"])
                + F.mse_loss(outputs["tran_morph_to_rna"], batch_dict["rna"])
                + F.mse_loss(outputs["tran_morph_to_ephys"], batch_dict["ephys"])
            )

            # 3. 最优传输对齐
            cost_re, _ = build_hybrid_cost(
                outputs["z_rna"], outputs["z_ephys"],
                model.cluster_layer_rna, model.cluster_layer_ephys,
                lambda_semantic=0.3,
            )
            cost_rm, _ = build_hybrid_cost(
                outputs["z_rna"], outputs["z_morph"],
                model.cluster_layer_rna, model.cluster_layer_morph,
                lambda_semantic=0.3,
            )
            loss_ot = ot_layer(cost_re)["ot_loss"] + ot_layer(cost_rm)["ot_loss"]

            # 4. 三模态 DEC 自训练聚类损失
            loss_cluster = (
                model.compute_dec_loss(outputs["q_rna"])
                + model.compute_dec_loss(outputs["q_ephys"])
                + model.compute_dec_loss(outputs["q_morph"])
            ) / 3.0

            total_loss = (
                ft_lambda_rec * loss_rec
                # + ft_lambda_tran * loss_tran
                + ft_lambda_ot * loss_ot
                + ft_lambda_cluster * loss_cluster
            )

            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["rec"] += loss_rec.item()
            epoch_losses["tran"] += loss_tran.item()
            epoch_losses["ot"] += loss_ot.item()
            epoch_losses["cluster"] += loss_cluster.item()

        n_batches = len(all_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        # 评估验证（使用原生三模态融合 Logits 判别）

        # 评估验证 (提取全量拼接表征后跑 KMeans)
        if epoch % 2 == 0 or epoch == finetune_epochs:
            model.eval()
            z_r_list, z_e_list, z_m_list = [], [], []
            all_true = []

            with torch.no_grad():
                for batch_dict in all_loader:
                    batch_dict = move_batch_to_device(batch_dict, device)
                    out = model(batch_dict)

                    z_r_list.append(out["z_rna"].cpu().numpy())
                    z_e_list.append(out["z_ephys"].cpu().numpy())
                    z_m_list.append(out["z_morph"].cpu().numpy())

                    if "labels" in batch_dict and "fine" in batch_dict["labels"]:
                        all_true.append(batch_dict["labels"]["fine"].cpu().numpy())

            # 1. 拼接三模态特征构成联合表征 [num_cells, latent_dim * 3]
            z_rna = np.vstack(z_r_list)
            z_ephys = np.vstack(z_e_list)
            z_morph = np.vstack(z_m_list)
            cur_z = np.concatenate([z_rna, z_ephys, z_morph], axis=1)

            # 2. 使用 KMeans 进行聚类判别预测
            val_kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=42)
            cur_preds = val_kmeans.fit_predict(cur_z)

            cur_labels = np.concatenate(all_true) if len(all_true) > 0 else None

            if cur_labels is not None:
                cur_ari = adjusted_rand_score(cur_labels, cur_preds)
                cur_nmi = normalized_mutual_info_score(cur_labels, cur_preds)
                rec_loss_sum = avg_losses["rec"]
                print(
                    f"⭐ Fine-tune Epoch [{epoch:03d}/{finetune_epochs}] "
                    f"| Loss: {avg_losses['total']:.4f} (Rec: {rec_loss_sum:.3f}, DEC: {avg_losses['cluster']:.4f}) "
                    f"| KMeans ARI: {cur_ari:.4f} | KMeans NMI: {cur_nmi:.4f}"
                )

                if cur_ari > best_ari:
                    best_ari = cur_ari
                    best_nmi = cur_nmi
                    torch.save(model.state_dict(), best_model_path)

                    # 导出拼接后的特征矩阵 [num_cells, latent_dim * 3]
                    latent_cols = [f"z_{i}" for i in range(cur_z.shape[1])]
                    df_latent = pd.DataFrame(cur_z, columns=latent_cols)
                    df_latent["true_label"] = cur_labels
                    df_latent["kmeans_pred"] = cur_preds

                    best_latent_csv_path = os.path.join(save_dir, "patchseq_best_latent_z.csv")
                    df_latent.to_csv(best_latent_csv_path, index=False)

                    print(f"  🏆 模型性能创新高 (KMeans ARI: {best_ari:.4f})，已保存权重至: {best_model_path}")
                    print(f"  📄 隐特征已同步导出至: {best_latent_csv_path}")
            else:
                print(
                    f"⭐ Fine-tune Epoch [{epoch:03d}/{finetune_epochs}] | Loss: {avg_losses['total']:.4f}"
                )


    print(f"\n✅ 训练全部完成！最优微调模型 ARI: {best_ari:.4f}, NMI: {best_nmi:.4f}")
    return model




if __name__ == "__main__":
    trained_model = train()





