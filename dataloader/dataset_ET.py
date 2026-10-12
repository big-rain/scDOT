
import os

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class PatchSeqBatchDataset(Dataset):
    def __init__(self,tensors_dict, labels_dict=None, pair_known_dict=None):
        if not tensors_dict:
            raise ValueError("tensors_dict 不能为空")

        self.tensors = tensors_dict
        self.labels = labels_dict
        self.pair_known = pair_known_dict
        self.n_samples = next(iter(self.tensors.values())).shape[0]

        # 检查各模态样本数。
        for mod, tensor in self.tensors.items():
            if tensor.shape[0] != self.n_samples:
                raise ValueError(f"{mod} 的样本数与其他模态不一致: {tensor.shape[0]} != {self.n_samples}")

        # 检查标签样本数。
        if self.labels is not None:
            for key, tensor in self.labels.items():
                if tensor.shape[0] != self.n_samples:
                    raise ValueError( f"标签 {key} 的样本数不一致: {tensor.shape[0]} != {self.n_samples}")

        # 检查 pair-known mask。
        if self.pair_known is not None:
            if set(self.pair_known.keys()) != set(self.tensors.keys()):
                raise ValueError("pair_known_dict 必须覆盖所有模态")

            for mod, mask in self.pair_known.items():
                if mask.ndim != 1:
                    raise ValueError(f"{mod}_pair_known 必须是一维 BoolTensor")

                if mask.shape[0] != self.n_samples:
                    raise ValueError(f"{mod}_pair_known 长度不匹配: {mask.shape[0]} != {self.n_samples}")

                if mask.dtype != torch.bool:
                    raise TypeError(f"{mod}_pair_known 必须是 torch.bool")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {mod: tensor[idx] for mod, tensor in self.tensors.items()}

        if self.labels is not None:
            sample["labels"] = {key: value[idx] for key, value in self.labels.items()}

        if self.pair_known is not None:
            for mod, mask in self.pair_known.items():
                sample[f"{mod}_pair_known"] = mask[idx]

        return sample

def build_patchseq_data(
    rna_input,
    ephys_input,
    morph_input,
    fine_label_key="label",
    batch_size=512,
    device="cpu",
    pair_mask_modality=None,
    pair_mask_rate=0.0,
    pair_mask_rates=None,
    pair_mask_seed=42,
):
    modality_names = ("rna", "ephys", "morph")

    # 1. 读取 AnnData
    def _ensure_adata(inp):
        if isinstance(inp, (str, os.PathLike)):
            return ad.read_h5ad(inp)
        return inp

    adatas = {
        "rna": _ensure_adata(rna_input),
        "ephys": _ensure_adata(ephys_input),
        "morph": _ensure_adata(morph_input),
    }

    # 2. 检查三种模态的细胞顺序是否一致
    obs_names = adatas["rna"].obs_names
    n_cells = len(obs_names)

    if n_cells == 0:
        raise ValueError("输入数据不包含任何细胞")

    if not obs_names.is_unique:
        raise ValueError("RNA 的 obs_names 必须唯一")

    for mod in ("ephys", "morph"):
        if not np.array_equal( obs_names,adatas[mod].obs_names,):
            raise ValueError(f"RNA 与 {mod} 的细胞顺序或数量不一致")

    feature_names = {mod: adata.var_names.tolist() for mod, adata in adatas.items()}

    # 3. 生成 pair-known mask 配置
    if pair_mask_rates is not None:
        if pair_mask_modality is not None:
            raise ValueError("pair_mask_rates 不能与 pair_mask_modality 同时使用")

        if pair_mask_rate != 0.0:
            raise ValueError("pair_mask_rates 不能与 pair_mask_rate 同时使用")

        unknown_modalities = (set(pair_mask_rates.keys()) - set(modality_names))

        if unknown_modalities:
            raise ValueError(f"pair_mask_rates 包含未知模态: {sorted(unknown_modalities)}")

        mask_rates = { mod: float(pair_mask_rates.get(mod, 0.0)) for mod in modality_names}

    else:
        mask_rates = {mod: 0.0 for mod in modality_names}

        if pair_mask_modality is None:
            if pair_mask_rate != 0.0:
                raise ValueError("设置 pair_mask_rate 时必须指定 pair_mask_modality")
        else:
            if pair_mask_modality not in modality_names:
                raise ValueError(f"pair_mask_modality 必须是以下之一: {modality_names}")

            mask_rates[pair_mask_modality] = float(pair_mask_rate)

    for mod, rate in mask_rates.items():
        if not np.isfinite(rate):
            raise ValueError( f"{mod} 的 mask 比例必须是有限数值")

        if not 0.0 <= rate <= 1.0:
            raise ValueError( f"{mod} 的 mask 比例必须在 [0, 1] 内，实际值为 {rate}")

    # 4. 生成固定的 pair-known mask
    # 每个模态使用独立随机流。
    seed_sequences = np.random.SeedSequence(pair_mask_seed).spawn(len(modality_names))
    pair_known_np = {}
    for mod, seed_sequence in zip(modality_names,seed_sequences):
        rng = np.random.default_rng(seed_sequence)

        # True 表示该模态的细胞对应关系可用于训练。
        mask = np.ones(n_cells, dtype=bool)
        n_hidden = int(round(n_cells * mask_rates[mod]))

        if n_hidden > 0:
            hidden_idx = rng.choice(n_cells,size=n_hidden,replace=False)
            mask[hidden_idx] = False

        pair_known_np[mod] = mask

    # =========================================================
    # 5. 处理标签
    # =========================================================
    obs = adatas["rna"].obs
    if fine_label_key not in obs.columns:
        raise ValueError(f"找不到细粒度标签列: {fine_label_key}")

    fine_values = (obs[fine_label_key].astype(str).to_numpy())
    coarse_values = np.asarray([label.split("-")[0] for label in fine_values],dtype=str,)
    fine_classes = sorted(np.unique(fine_values).tolist())
    coarse_classes = sorted(np.unique(coarse_values).tolist())

    fine_map = {label: index for index, label in enumerate(fine_classes)}
    coarse_map = {label: index for index, label in enumerate(coarse_classes)}

    fine_encoded = np.asarray([fine_map[label] for label in fine_values],dtype=np.int64)
    coarse_encoded = np.asarray([coarse_map[label] for label in coarse_values],dtype=np.int64)
    fine_to_coarse_idx = np.asarray([coarse_map[label.split("-")[0]] for label in fine_classes],dtype=np.int64)

    # 6. 标准化并转换为 Tensor
    # 不进行 train/val 划分，因此 scaler 在全部细胞上拟合。
    tensors_all = {}
    scalers = {}

    for mod, adata in adatas.items():
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] != n_cells:
            raise ValueError(f"{mod} 的样本数不一致: {X.shape[0]} != {n_cells}" )

        scaler = StandardScaler()
        scaler.fit(X)
        X_norm = scaler.transform(X).astype(np.float32, copy=False)
        tensors_all[mod] = torch.as_tensor( X_norm,dtype=torch.float32,  device=device)
        scalers[mod] = scaler

    labels_all = {
        "fine": torch.as_tensor(fine_encoded,dtype=torch.long, device=device,),
        "coarse": torch.as_tensor(coarse_encoded,dtype=torch.long, device=device,),
    }

    pair_known_all = { mod: torch.as_tensor(mask,dtype=torch.bool,device=device,) for mod, mask in pair_known_np.items()}

    # =========================================================
    # 7. 创建 Dataset 和 DataLoader
    # =========================================================
    dataset = PatchSeqBatchDataset(
        tensors_dict=tensors_all,
        labels_dict=labels_all,
        pair_known_dict=pair_known_all,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # =========================================================
    # 8. 返回数据
    # =========================================================
    return {
        "all": {
            "tensors": tensors_all,
            "labels": labels_all,
            "pair_known": pair_known_all,
            "obs_names": obs_names.tolist(),
            "dataset": dataset,
            "loader": loader,
        },
        "meta": {
            "obs_names": obs_names.tolist(),
            "feature_names": feature_names,
            "fine_classes": fine_classes,
            "coarse_classes": coarse_classes,
            "fine_to_index": fine_map,
            "coarse_to_index": coarse_map,
            "fine_to_coarse_idx": fine_to_coarse_idx,

            # 用户设置的隐藏比例。
            "pair_mask_rates": mask_rates,

            # 用于复现实验。
            "pair_mask_seed": pair_mask_seed,

            # 保存全量细胞上的实际 mask。
            "pair_known": {
                mod: mask.tolist()
                for mod, mask in pair_known_np.items()
            },

            # 实际隐藏比例，通常与设置值非常接近。
            "pair_mask_actual_rates": {
                mod: float((~mask).mean())
                for mod, mask in pair_known_np.items()
            },
        },
        "scalers": scalers,
    }