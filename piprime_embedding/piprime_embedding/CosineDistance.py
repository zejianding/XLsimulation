import numpy as np
from openpyxl import Workbook


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=r"QLEEAEEEAQR_28.npz")
    parser.add_argument("--output", default=r"QLEEAEEEAQR_28.xlsx")
    args = parser.parse_args()
    npz_path = args.npz
    excel_path = args.output

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]  # shape: [N, D]

    n = embeddings.shape[0]

    if n == 0:
        print("NPZ 中没有 embeddings")
        return

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = embeddings / norms

    # cosine similarity matrix
    sim_matrix = normalized @ normalized.T

    # 计算每一行去掉自己的平均值
    row_means_excluding_self = []
    for i in range(n):
        if n == 1:
            mean_val = 0.0
        else:
            row_sum = np.sum(sim_matrix[i]) - sim_matrix[i, i]  # 去掉自己
            mean_val = row_sum / (n - 1)
        row_means_excluding_self.append(float(mean_val))

    wb = Workbook()
    ws = wb.active
    ws.title = "cosine_similarity"

    # 第一行表头
    header = [""] + list(range(n)) + ["mean_excluding_self"]
    ws.append(header)

    # 写入矩阵 + 平均值
    for i in range(n):
        row = [i] + sim_matrix[i].tolist() + [row_means_excluding_self[i]]
        ws.append(row)

    wb.save(excel_path)
    print(f"已保存: {excel_path}")
    print(f"矩阵大小: {sim_matrix.shape}")


if __name__ == "__main__":
    main()