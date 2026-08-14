import numpy as np

from graph import tools


num_node = 13

self_link = [
    (i, i) for i in range(num_node)
]

# Penn Action 关键点定义：
#
# 0: head
# 1: right shoulder
# 2: left shoulder
# 3: right elbow
# 4: left elbow
# 5: right wrist
# 6: left wrist
# 7: right hip
# 8: left hip
# 9: right knee
# 10: left knee
# 11: right ankle
# 12: left ankle
#
# 所有索引均为 Python 0-based 索引。
inward = [
    (0, 1),    # head -> right shoulder
    (0, 2),    # head -> left shoulder

    (1, 3),    # right shoulder -> right elbow
    (3, 5),    # right elbow -> right wrist

    (2, 4),    # left shoulder -> left elbow
    (4, 6),    # left elbow -> left wrist

    (1, 7),    # right shoulder -> right hip
    (2, 8),    # left shoulder -> left hip

    (7, 9),    # right hip -> right knee
    (9, 11),   # right knee -> right ankle

    (8, 10),   # left hip -> left knee
    (10, 12),  # left knee -> left ankle

    (7, 8),    # right hip -> left hip
]

outward = [
    (j, i) for i, j in inward
]

neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode='spatial'):
        if labeling_mode == 'spatial':
            return tools.get_spatial_graph(
                num_node=num_node,
                self_link=self_link,
                inward=inward,
                outward=outward
            )

        raise ValueError(
            f'Unsupported labeling mode: {labeling_mode}'
        )
