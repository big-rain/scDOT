import os
import numpy as np
from copy import deepcopy
from numpy.random import choice
from numpy import setdiff1d
from anndata import AnnData
from scipy.sparse import csr_matrix
import scipy.io as sio



def write_adata(adata, folder_path, modal, feature, feature_name=None, transpose=True):
    """
    导出 AnnData 为 mtx 矩阵文件及对应的 features.tsv / barcodes.tsv
    """
    import scipy.io as sio
    import scipy.sparse as sp

    os.makedirs(folder_path, exist_ok=True)

    # 1. 保存表达量矩阵为 mtx 格式
    matrix = adata.X.T if transpose else adata.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    sio.mmwrite(os.path.join(folder_path, "matrix.mtx"), matrix)

    # 2. 保存细胞条形码 (Barcodes)
    pd.Series(adata.obs_names).to_csv(
        os.path.join(folder_path, "barcodes.tsv"),
        sep="\t", header=False, index=False
    )

    # 3. 保存基因 / Peak 特征列表 (Features / Genes)
    # 优先从 var 中找指定的 feature_name 列，如果找不到或未指定，直接用 var_names (Index)
    if feature_name and feature_name in adata.var.columns:
        features_df = adata.var[[feature_name]]
    elif feature_name and feature_name in adata.obs.columns:
        features_df = adata.obs[[feature_name]]
    else:
        features_df = pd.DataFrame(index=adata.var_names)

    features_df.to_csv(
        os.path.join(folder_path, f"{feature}.tsv"),
        sep="\t", header=False, index=True
    )

def delete_symlink(path):
    """安全删除符号链接或文件"""
    if os.path.exists(path) or os.path.islink(path):
        try:
            os.remove(path)
            return True
        except Exception:
            return False
    return False

def downsample_samples(adata_p1, adata_p2, adata_u1, adata_u2, depth_multi, depth_urna, depth_uatac,
                       downsample_fragment=True, parent_dir=None, frag_path=""):
    import utils_eval
    import pandas as pd
    import os

    assert depth_multi > 0 and depth_multi <= 1, "please input depth_multi in the correct range (0,1]"
    assert depth_urna > 0 and depth_urna <= 1, "please input depth_urna in the correct range (0,1]"
    assert depth_uatac > 0 and depth_uatac <= 1, "please input depth_uatac in the correct range (0,1]"

    # ==== RNA portion (run function if ratio != 1) ====
    if depth_multi != 1:
        print("downsampling paired RNA")
        adata_p1.X = utils_eval.downsample_rna(adata_p1.X.todense().T, depth_multi, bycol=True).T
    if depth_urna != 1:
        print("downsampling unpaired RNA")
        adata_u1.X = utils_eval.downsample_rna(adata_u1.X.todense().T, depth_urna).T

    # ==== ATAC portion ====
    if downsample_fragment:
        if parent_dir is None:
            print("please specify where the fragment files should be saved")
            return
        else:
            pd.DataFrame(adata_p2.var_names).to_csv("tmp.csv", header=False, index=False)
            if depth_multi < 1:
                print("downsampling paired ATAC")
                print("deleting previously written symlink file(fragments.tsv.gz)")
                pfolder = parent_dir + "/paired_ATAC/"
                if delete_symlink(os.path.join(pfolder, "fragments.tsv.gz")):
                    os.remove(os.path.join(pfolder, "fragments.tsv.gz"))
                if delete_symlink(os.path.join(pfolder, "fragments.tsv.gz.tbi")):
                    os.remove(os.path.join(pfolder, "fragments.tsv.gz.tbi"))

                adata_p2.X = utils_eval.downsample_atac(list(adata_p2.obs.index),
                                                        depth_multi,
                                                        os.path.join(parent_dir, "paired_ATAC"),
                                                        frag_path=frag_path,
                                                        feature_path="tmp.csv").T
            if depth_uatac < 1:
                print("downsampling unpaired ATAC")
                print("deleting previously written symlink file(fragments.tsv.gz)")
                pfolder = os.path.join(parent_dir, "unpaired_ATAC")
                if delete_symlink(os.path.join(pfolder, "fragments.tsv.gz")):
                    os.remove(os.path.join(pfolder, "fragments.tsv.gz"))
                if delete_symlink(os.path.join(pfolder, "fragments.tsv.gz.tbi")):
                    os.remove(os.path.join(pfolder, "fragments.tsv.gz.tbi"))
                adata_u2.X = utils_eval.downsample_atac(list(adata_u2.obs.index),
                                                        depth_uatac,
                                                        os.path.join(parent_dir, "unpaired_ATAC"),
                                                        frag_path=frag_path,
                                                        feature_path="tmp.csv").T
            os.remove("tmp.csv")
    else:
        if depth_multi != 1:
            print("downsampling paired ATAC")
            adata_p2.X = utils_eval.downsample_rna(adata_p2.X.todense().T, depth_multi, bycol=True).T
        if depth_uatac != 1:
            print("downsampling unpaired ATAC")
            adata_u2.X = utils_eval.downsample_rna(adata_u2.X.todense().T, depth_uatac).T

    return (adata_p1, adata_p2, adata_u1, adata_u2)



def pair_unpair_split_size(
        adata_rna,
        adata_atac,
        n_multi,
        n_urna=None,
        n_uatac=None,
        transpose=False,
        fragment_path=None,
        parent_dir=None
):
    """
    内存安全版的全量 / 混合配对切分函数 (Fixed Memory-Safe Version)
    实现 paired_idx, u1_idx (unpaired RNA), u2_idx (unpaired ATAC) 的三方互斥切分
    """
    subsample_size = lambda vector, size: choice(vector, size=size, replace=False)

    if transpose:
        adata_rna = adata_rna.copy().transpose()
        adata_atac = adata_atac.copy().transpose()

    assert adata_rna.shape[0] == adata_atac.shape[0], "RNA/ATAC 细胞数量不一致！"

    ncells = adata_rna.shape[0]
    all_indices = np.arange(ncells)

    # 1. 抽取配对样本索引
    paired_idx = subsample_size(all_indices, n_multi)
    unpaired_idx = setdiff1d(all_indices, paired_idx)

    # 2. 计算并抽取非配对样本索引
    if n_urna is None and n_uatac is None:
        n_urna = round(len(unpaired_idx) / 2)
        n_uatac = len(unpaired_idx) - n_urna
    elif n_urna is not None and n_uatac is None:
        n_uatac = len(unpaired_idx) - n_urna
    elif n_urna is None and n_uatac is not None:
        n_urna = len(unpaired_idx) - n_urna
    elif (n_urna + n_uatac + n_multi) > ncells:
        raise ValueError("请求的细胞总数超过了 AnnData 中可用的细胞总数！")

    u1_idx = subsample_size(unpaired_idx, n_urna)
    remain_idx = setdiff1d(unpaired_idx, u1_idx)
    u2_idx = subsample_size(remain_idx, n_uatac)

    # 3. 内存安全地创建 4 个 AnnData 实例 (避免使用会导致 OOM 的 .todense())
    adata_p1 = AnnData(
        X=csr_matrix(adata_rna.X[paired_idx, :]),
        obs=adata_rna.obs.iloc[paired_idx, :].copy(),
        var=adata_rna.var.copy(),
        dtype=np.float32
    )
    adata_p2 = AnnData(
        X=csr_matrix(adata_atac.X[paired_idx, :]),
        obs=adata_atac.obs.iloc[paired_idx, :].copy(),
        var=adata_atac.var.copy(),
        dtype=np.float32
    )
    adata_u1 = AnnData(
        X=csr_matrix(adata_rna.X[u1_idx, :]),
        obs=adata_rna.obs.iloc[u1_idx, :].copy(),
        var=adata_rna.var.copy(),
        dtype=np.float32
    )
    adata_u2 = AnnData(
        X=csr_matrix(adata_atac.X[u2_idx, :]),
        obs=adata_atac.obs.iloc[u2_idx, :].copy(),
        var=adata_atac.var.copy(),
        dtype=np.float32
    )

    if transpose:
        adata_p1 = adata_p1.copy().transpose()
        adata_p2 = adata_p2.copy().transpose()
        adata_u1 = adata_u1.copy().transpose()
        adata_u2 = adata_u2.copy().transpose()

    # 4. Fragment 文件的软链接建立 (适用于 ATAC-seq 分析)
    if (fragment_path is not None) and (parent_dir is not None):
        tbi_file = fragment_path + ".tbi"
        if os.path.exists(fragment_path) and os.path.exists(tbi_file):
            pfolder = os.path.join(parent_dir, "paired_ATAC")
            ufolder = os.path.join(parent_dir, "unpaired_ATAC")
            os.makedirs(pfolder, exist_ok=True)
            os.makedirs(ufolder, exist_ok=True)

            delete_symlink(os.path.join(pfolder, "fragments.tsv.gz"))
            delete_symlink(os.path.join(pfolder, "fragments.tsv.gz.tbi"))
            delete_symlink(os.path.join(ufolder, "fragments.tsv.gz"))
            delete_symlink(os.path.join(ufolder, "fragments.tsv.gz.tbi"))

            os.symlink(fragment_path, os.path.join(pfolder, "fragments.tsv.gz"))
            os.symlink(tbi_file, os.path.join(pfolder, "fragments.tsv.gz.tbi"))
            os.symlink(fragment_path, os.path.join(ufolder, "fragments.tsv.gz"))
            os.symlink(tbi_file, os.path.join(ufolder, "fragments.tsv.gz.tbi"))

    return adata_p1, adata_p2, adata_u1, adata_u2


def data_simulation(
        in_dir,
        adata_rna,
        adata_atac,
        iter_list,
        depth_multiome_list,
        depth_scrna_list,
        depth_snatac_list,
        n_multiome_list,
        n_scrna_list,
        n_snatac_list,
        repeats,
        fragment_path,
        cond_key,
        fn,
        downsample=False,
        target_label_col='cell_type'
):
    """
    自动化数据模拟生成主循环
    包含 h5ad/mtx 数据导出以及【4 组独立 cell_types.tsv 标签全量导出】
    """
    for i in range(len(iter_list)):
        s_i = iter_list[i]
        for j in range(1, repeats + 1):
            folder_path = "{}{}{}_{}/".format(in_dir, cond_key, fn(s_i), str(j))
            print("=" * 80)
            print(f"Splitting dataset iteration {s_i}, repeat {j}")
            print(f"Output folder: {folder_path}")
            print("=" * 80)

            # 1. 划分数据集
            adata_p1, adata_p2, adata_u1, adata_u2 = pair_unpair_split_size(
                adata_rna,
                adata_atac,
                n_multi=n_multiome_list[i],
                n_urna=n_scrna_list[i],
                n_uatac=n_snatac_list[i],
                transpose=False,
                fragment_path=fragment_path,
                parent_dir=folder_path
            )

            # 2. 可选降采样 (Downsampling)
            if downsample:
                # 假设你已有 downsample_samples 函数定义
                adata_p1, adata_p2, adata_u1, adata_u2 = downsample_samples(
                    adata_p1, adata_p2, adata_u1, adata_u2,
                    depth_multi=depth_multiome_list[i],
                    depth_urna=depth_scrna_list[i],
                    depth_uatac=depth_snatac_list[i],
                    downsample_fragment=True,
                    parent_dir=folder_path,
                    frag_path=fragment_path
                )

            # 3. 写入单细胞表达量数据 (调用 utils_eval 工具库)
            # 假设 utils_eval.write_adata 已在你的工程环境中导入
            write_adata(adata_p1, os.path.join(folder_path, "paired_RNA"), "RNA", "gene",
                                   feature_name='feature', transpose=True)
            write_adata(adata_p2, os.path.join(folder_path, "paired_ATAC"), "ATAC", "peak",
                                   feature_name='feature', transpose=True)
            write_adata(adata_u1, os.path.join(folder_path, "unpaired_RNA"), "RNA", "gene",
                                   feature_name='feature', transpose=True)
            write_adata(adata_u2, os.path.join(folder_path, "unpaired_ATAC"), "ATAC", "peak",
                                   feature_name='feature', transpose=True)

            # 4. 【补全的核心部分】：为 4 个独立文件夹全面保存对应的 cell_types.tsv 标签
            if target_label_col in adata_rna.obs.columns:
                # ① Paired RNA 组
                if adata_p1.shape[0] > 0:
                    adata_p1.obs[[target_label_col]].to_csv(
                        os.path.join(folder_path, "paired_RNA/cell_types.tsv"), sep="\t"
                    )
                # ② Paired ATAC 组
                if adata_p2.shape[0] > 0:
                    adata_p2.obs[[target_label_col]].to_csv(
                        os.path.join(folder_path, "paired_ATAC/cell_types.tsv"), sep="\t"
                    )
                # ③ Unpaired RNA 组
                if adata_u1.shape[0] > 0:
                    adata_u1.obs[[target_label_col]].to_csv(
                        os.path.join(folder_path, "unpaired_RNA/cell_types.tsv"), sep="\t"
                    )
                # ④ Unpaired ATAC 组
                if adata_u2.shape[0] > 0:
                    adata_u2.obs[[target_label_col]].to_csv(
                        os.path.join(folder_path, "unpaired_ATAC/cell_types.tsv"), sep="\t"
                    )

                print(
                    f"✅ [Label Export OK] 成功为 paired/unpaired 各组导出 cell_types.tsv (标签列: {target_label_col!r})")
            else:
                print(f"⚠️ [Warning] 在原始数据的 obs 中未找到 {target_label_col!r} 列，无法生成 cell_types.tsv！")


if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    import anndata as ad
    from copy import deepcopy
    from scipy.sparse import csr_matrix

    # 1. 基础路径配置
    source_dir = r"D:\PycharmProjects\C1\dataset\PEMC10kpublic"
    in_dir = r"D:\PycharmProjects\C1\dataset\PEMC10kpublic/"

    print("⏳ 正在读取原始 AnnData 数据...")
    adata_rna = ad.read_h5ad(os.path.join(source_dir, "pbmc_10x_rna_public.h5ad"))
    adata_atac = ad.read_h5ad(os.path.join(source_dir, "pbmc_10x_atac_public.h5ad"))

    total_cells = adata_rna.shape[0]
    print(f"✅ 数据读取完成！总细胞数: {total_cells}")

    # 2. 文件夹命名标识
    cond_key = "paired_ratio_"

    # =========================================================================
    # 🎯 科学设计的 4 大核心梯度 (0% -> 20% -> 50% -> 80%)
    #  - ratio_0pct  (0% Paired):   0 paired, ~5000 uRNA, ~5000 uATAC (纯非配对对齐)
    #  - ratio_20pct (20% Paired): ~2000 paired, ~4000 uRNA, ~4000 uATAC (极低配对辅助)
    #  - ratio_50pct (50% Paired): ~5000 paired, ~2500 uRNA, ~2500 uATAC (标准半配对)
    #  - ratio_80pct (80% Paired): ~8000 paired, ~1000 uRNA, ~1000 uATAC (高配对基准)
    # =========================================================================

    # 实验组标识名称
    iter_list = ["ratio_0pct", "ratio_20pct", "ratio_50pct", "ratio_80pct", 'ratio_100pct']

    # 对应的配对细胞数 (n_multiome)
    n_multiome_list = [
        int(total_cells * 0.0),  # 0 (全非配对)
        int(total_cells * 0.2),  # ~2000
        int(total_cells * 0.5),  # ~5000
        int(total_cells * 0.8),  # ~8000
        int(total_cells )        #  全配对
    ]

    # 剩余的细胞平分给 unpaired RNA 和 unpaired ATAC
    # 剩余细胞平分：强制让 n_scrna 和 n_snatac 的数量绝对相等
    n_scrna_list = [
        (total_cells - n_m) // 2 for n_m in n_multiome_list
    ]

    # 强行让 ATAC 数量与 RNA 严格相等
    n_snatac_list = n_scrna_list.copy()

    # 测序深度默认保持 1.0
    depth_multiome_list = [1.0] * len(iter_list)
    depth_scrna_list = [1.0] * len(iter_list)
    depth_snatac_list = [1.0] * len(iter_list)

    # 重复次数 (先设为 1，确认无误后再设为 3~5 跑多次重复验证)
    repeats = 1

    # 3. Fragment 文件路径控制
    fragment_path = os.path.join(source_dir, "pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz")
    if not os.path.exists(fragment_path):
        print(f"⚠️ 未找到 fragment 文件: {fragment_path}，将禁用符号链接。")
        fragment_path = None


    def to_str(s):
        return str(s)


    print("\n🚀 开启多梯度数据切分与模拟导出 (包含全非配对组)...")
    for idx, (ratio_name, n_p, n_ur, n_ua) in enumerate(zip(iter_list, n_multiome_list, n_scrna_list, n_snatac_list)):
        print(f"  🔹 [{ratio_name:<11}] Paired: {n_p:<5} | Unpaired RNA: {n_ur:<5} | Unpaired ATAC: {n_ua:<5}")

    # 4. 调用模拟生成主函数
    data_simulation(
        in_dir=in_dir,
        adata_rna=adata_rna,
        adata_atac=adata_atac,
        iter_list=iter_list,
        depth_multiome_list=depth_multiome_list,
        depth_scrna_list=depth_scrna_list,
        depth_snatac_list=depth_snatac_list,
        n_multiome_list=n_multiome_list,
        n_scrna_list=n_scrna_list,
        n_snatac_list=n_snatac_list,
        repeats=repeats,
        fragment_path=fragment_path,
        cond_key=cond_key,
        fn=to_str,
        downsample=False,
        target_label_col='ct3'
    )
    print("\n🎉 全梯度数据模拟生成完成！生成的 0%, 20%, 50%, 80% 组可直接用于论文绘图对比。")