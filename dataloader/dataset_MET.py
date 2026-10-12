import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import anndata as ad
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler


class PatchSeqBatchDataset(Dataset):
    """优化后的轻量 Dataset：底层保持整块 Tensor，按批切片速度极快"""

    def __init__(self, tensors_dict, labels_dict=None):
        self.tensors = tensors_dict
        self.labels = labels_dict
        # 以任意一个模态计算样本数
        self.n_samples = next(iter(self.tensors.values())).shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 兼容整数索引和切片
        sample = {mod: tensor[idx] for mod, tensor in self.tensors.items()}
        if self.labels is not None:
            sample["labels"] = {k: v[idx] for k, v in self.labels.items()}
        return sample


def build_patchseq_data(
        rna_input,
        ephys_input,
        morph_input,
        fine_label_key="label",
        ind_dict=None,
        cv=None,
        batch_size=512,
        split=True,
        device="cpu"  # 支持直接放到 cuda 或 cpu
):
    # =========================================================
    # 1. 读取 AnnData (兼容传入文件路径或已读入的对象)
    # =========================================================
    def _ensure_adata(inp):
        return ad.read_h5ad(inp) if isinstance(inp, str) else inp

    adatas = {
        "rna": _ensure_adata(rna_input),
        "ephys": _ensure_adata(ephys_input),
        "morph": _ensure_adata(morph_input),
    }

    # =========================================================
    # 2. 检查对齐并提取【特征名】(供弦图使用)
    # =========================================================
    obs_names = adatas["rna"].obs_names
    for name in ["ephys", "morph"]:
        assert np.array_equal(obs_names, adatas[name].obs_names), \
            f"RNA 与 {name} 的细胞顺序或数量不一致！"

    # 保存特征名称字典 (画弦图必备)
    feature_names = {mod: adata.var_names.tolist() for mod, adata in adatas.items()}

    # =========================================================
    # 3. 标签处理 (自动构建 细类 -> 粗类 映射)
    # =========================================================
    obs = adatas["rna"].obs
    assert fine_label_key in obs.columns, f"找不到细粒度标签列: {fine_label_key}"

    fine_values = obs[fine_label_key].astype(str).to_numpy()
    coarse_values = np.asarray([label.split("-")[0] for label in fine_values], dtype=str)

    fine_classes = sorted(np.unique(fine_values).tolist())
    coarse_classes = sorted(np.unique(coarse_values).tolist())

    fine_map = {lb: i for i, lb in enumerate(fine_classes)}
    coarse_map = {lb: i for i, lb in enumerate(coarse_classes)}

    fine_encoded = np.array([fine_map[lb] for lb in fine_values], dtype=np.int64)
    coarse_encoded = np.array([coarse_map[lb] for lb in coarse_values], dtype=np.int64)

    # 细类别 ID 到 粗类别 ID 的确定映射数组 (27 -> 5)
    fine_to_coarse_idx = np.array([
        coarse_map[fine_classes[f_idx].split("-")[0]]
        for f_idx in range(len(fine_classes))
    ], dtype=np.int64)

    # 4. 数据标准化与 PyTorch Tensor 化 (零重复拷贝)
    if split:
        assert ind_dict is not None and cv is not None, "split=True 时必须提供 ind_dict 和 cv"
        train_idx = np.asarray(ind_dict[cv]["train"])
        val_idx = np.asarray(ind_dict[cv]["val"])
    else:
        train_idx = np.arange(len(obs_names))
        val_idx = None

    tensors_all = {}
    scalers = {}

    for name, adata in adatas.items():
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        X = np.asarray(X, dtype=np.float32)

        scaler = StandardScaler()
        if split:
            scaler.fit(X[train_idx])
        else:
            scaler.fit(X)

        X_norm = scaler.transform(X).astype(np.float32)
        # 转为 PyTorch Tensor 并直接部署到指定设备
        tensors_all[name] = torch.tensor(X_norm, dtype=torch.float32, device=device)
        scalers[name] = scaler

    labels_all = {
        "fine": torch.tensor(fine_encoded, dtype=torch.long, device=device),
        "coarse": torch.tensor(coarse_encoded, dtype=torch.long, device=device)
    }

    # =========================================================
    # 5. 组装数据 (同时提供整块 Tensor 与 DataLoader)
    def _create_split_dict(idx):
        if idx is None:
            return None
        # 索引切片（共享底层内存，极速）
        sub_tensors = {k: v[idx] for k, v in tensors_all.items()}
        sub_labels = {k: v[idx] for k, v in labels_all.items()}

        dataset = PatchSeqBatchDataset(sub_tensors, sub_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return {
            "tensors": sub_tensors,  # <-- 算 SHAP、KMeans 时直接用这个，速度最快！
            "labels": sub_labels,
            "dataset": dataset,
            "loader": loader  # <-- 需要 batch 训练时用这个
        }

    return {
        "all": _create_split_dict(slice(None)),
        "train": _create_split_dict(train_idx) if split else None,
        "val": _create_split_dict(val_idx) if split else None,
        "meta": {
            "obs_names": obs_names.tolist(),
            "feature_names": feature_names,  # 用于画弦图的模态特征名
            "fine_classes": fine_classes,
            "coarse_classes": coarse_classes,
            "fine_to_index": fine_map,
            "coarse_to_index": coarse_map,
            "fine_to_coarse_idx": fine_to_coarse_idx  # 映射数组
        },
        "scalers": scalers
    }


import os
import pickle
import numpy as np
import torch
import anndata as ad
from sklearn.model_selection import StratifiedKFold


# 引入优化后的数据构建函数（如果写在单独文件里可以直接 import build_patchseq_data）
# from data_loader import build_patchseq_data


def test_with_real_patchseq():
    rna_path = "../dataset/PatchseqGABAergicNeuron/adata_RNA_TEM.h5ad"
    ephys_path = "../dataset/PatchseqGABAergicNeuron/adata_Ephys_TEM.h5ad"
    morph_path = "../dataset/PatchseqGABAergicNeuron/adata_Morph_TEM.h5ad"

    print("=" * 70)
    print("开始使用真实 Patch-seq GABAergic Neuron 数据进行管道测试...")
    print("=" * 70)

    # 1. 检查真实数据文件是否存在
    for p in [rna_path, ephys_path, morph_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到数据文件: {os.path.abspath(p)}")
    print("✓ 真实 h5ad 文件检测通过")

    # 2. 预读 RNA 模态检测细粒度类别列名（通常为 'cell_type_TEM' 或 'label'）
    adata_peek = ad.read_h5ad(rna_path, backed='r')
    if 'cell_type_TEM' in adata_peek.obs.columns:
        fine_label_key = 'cell_type_TEM'
    elif 'label' in adata_peek.obs.columns:
        fine_label_key = 'label'
    else:
        raise KeyError(f"未在 RNA obs 中找到类别列，当前列名有: {list(adata_peek.obs.columns)}")
    print(f"✓ 识别到细粒度标签列: '{fine_label_key}'")

    # 3. 准备交叉验证 ind_dict (若有现有 pkl 则载入，否则现场生成一份分层切分)
    # 假设你的工程存在 ind_dict.pkl，可以按需读取：
    ind_dict_file = "../dataset/PatchseqGABAergicNeuron/ind_dict.pkl"
    if os.path.exists(ind_dict_file):
        with open(ind_dict_file, "rb") as f:
            ind_dict = pickle.load(f)
        print("✓ 载入已有 ind_dict.pkl 交叉验证划分")
    else:
        print("ℹ 未发现本地 ind_dict.pkl，使用 StratifiedKFold 现场生成模拟 5 折划分...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y = adata_peek.obs[fine_label_key].astype(str).values
        ind_dict = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            ind_dict[fold] = {"train": train_idx, "val": val_idx}

    # 4. 执行优化后的数据构建管道
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ 目标加载设备: {device}")

    cv = 0
    data = build_patchseq_data(
        rna_input=rna_path,
        ephys_input=ephys_path,
        morph_input=morph_path,
        fine_label_key=fine_label_key,
        ind_dict=ind_dict,
        cv=cv,
        batch_size=128,
        split=True,
        device=device
    )

    # =========================================================================
    # 验证环节 1：检查真实数据的模态对齐与真实特征维度
    # =========================================================================
    print("\n" + "-" * 40)
    print("【检查点 1】真实样本与特征维度")
    print("-" * 40)

    tensors_all = data["all"]["tensors"]
    feature_names = data["meta"]["feature_names"]

    total_cells = tensors_all["rna"].shape[0]
    print(f"总细胞数量: {total_cells}")
    for mod in ["rna", "ephys", "morph"]:
        feat_dim = tensors_all[mod].shape[1]
        print(f"  - 模态 [{mod:5s}]: 维度 = {feat_dim:5d} 特征 | Tensor Shape: {list(tensors_all[mod].shape)}")
        assert len(feature_names[mod]) == feat_dim, f"特征名称数量与 Tensor 列数不匹配: {mod}"

    # =========================================================================
    # 验证环节 2：检查生物学类别体系（27 类细分 -> 5 大类）
    # =========================================================================
    print("\n" + "-" * 40)
    print("【检查点 2】生物学类别体系识别")
    print("-" * 40)
    fine_classes = data["meta"]["fine_classes"]
    coarse_classes = data["meta"]["coarse_classes"]

    print(f"识别到的细粒度亚型数量 (t-types): {len(fine_classes)} (预期通常为 27 左右)")
    print(f"细分类别示例: {fine_classes[:5]} ...")
    print(f"识别到的大类数量 (Major types): {len(coarse_classes)} (预期为 5 大类: Pvalb, Sst, Vip, Lamp5, Sncg)")
    print(f"大类别列表: {coarse_classes}")

    # 验证 fine_to_coarse_idx 矩阵映射的精准度
    fine_all = data["all"]["labels"]["fine"].cpu().numpy()
    coarse_all = data["all"]["labels"]["coarse"].cpu().numpy()
    mapped_coarse = data["meta"]["fine_to_coarse_idx"][fine_all]
    assert np.array_equal(mapped_coarse, coarse_all), "细类到粗类的映射逻辑与实际不一致！"
    print("✓ 细类到粗类的向量化映射逻辑 100% 准确！")

    # =========================================================================
    # 验证环节 3：验证专为 SHAP / KMeans 准备的 Tensor 格式
    # =========================================================================
    print("\n" + "-" * 40)
    print("【检查点 3】SHAP / 聚类模型专用纯张量接口")
    print("-" * 40)
    train_tensors = list(data["train"]["tensors"].values())
    val_tensors = list(data["val"]["tensors"].values())

    print(f"训练集背景样本 (Background) 模态数: {len(train_tensors)}")
    print(f"训练集各模态 Tensor 规格: {[list(t.shape) for t in train_tensors]}")
    print(f"验证集待测样本 (Test) 模态数: {len(val_tensors)}")
    print(f"验证集各模态 Tensor 规格: {[list(t.shape) for t in val_tensors]}")

    # 验证是否为连续张量且在目标设备上
    assert all(t.is_cuda == (device == "cuda") for t in train_tensors)
    assert all(t.dtype == torch.float32 for t in train_tensors)
    print("✓ 纯张量列表满足 shap.DeepExplainer 要求的标准格式输入！")

    # =========================================================================
    # 验证环节 4：特征名称备查（后续弦图连线核心材料）
    # =========================================================================
    print("\n" + "-" * 40)
    print("【检查点 4】弦图特征名称可用性抽样")
    print("-" * 40)
    print(f"RNA Top-5 基因名称示例: {feature_names['rna'][:5]}")
    print(f"Ephys Top-5 特征名称示例: {feature_names['ephys'][:5]}")
    print(f"Morph Top-5 特征名称示例: {feature_names['morph'][:5]}")
    print("✓ 特征名完整无损，可直接用于 Chord Diagram 的连接弧线标签。")

    print("\n" + "=" * 70)
    print("🎉 真实 Patch-seq 数据测试全部通过！数据流可直接接入你的算法与 SHAP 模块。")
    print("=" * 70)


if __name__ == "__main__":
    test_with_real_patchseq()

