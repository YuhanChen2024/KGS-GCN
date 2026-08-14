import numpy as np


def edge2mat(link, num_node):
    """
    将边列表转换为邻接矩阵。

    Args:
        link: [(source, target), ...]
        num_node: 节点数量

    Returns:
        A: shape 为 (num_node, num_node) 的邻接矩阵
    """
    A = np.zeros((num_node, num_node), dtype=np.float32)

    for i, j in link:
        if not (0 <= i < num_node and 0 <= j < num_node):
            raise ValueError(
                f'Invalid edge ({i}, {j}) for graph with {num_node} nodes'
            )

        A[j, i] = 1.0

    return A


def normalize_digraph(A):
    """
    对有向图进行列归一化。
    """
    Dl = np.sum(A, axis=0)

    Dn = np.zeros_like(A, dtype=np.float32)

    for i in range(A.shape[1]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    return np.dot(A, Dn)


def get_spatial_graph(num_node, self_link, inward, outward):
    """
    构造空间图：

        A = [I, A_in, A_out]

    输出形状为：
        (3, num_node, num_node)
    """
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))

    return np.stack((I, In, Out)).astype(np.float32)
