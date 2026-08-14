import pickle

import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    """
    Penn Action 数据读取器。

    数据格式：
        data:  (N, C, T, V, M)
        label: (N,)

    其中：
        C: 坐标通道数，通常为 3，即 x、y、visibility
        T: 时间帧数
        V: 关键点数量，Penn Action 为 13
        M: 人数，Penn Action 为 1
    """

    def __init__(
        self,
        data_path,
        label_path,
        split='train',
        window_size=-1,
        debug=False,
        use_mmap=False,
        **kwargs
    ):
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.window_size = window_size
        self.debug = debug
        self.use_mmap = use_mmap

        self.load_data()

    def load_data(self):
        print(f'Loading data from {self.data_path} ...')

        mmap_mode = 'r' if self.use_mmap else None
        self.data = np.load(self.data_path, mmap_mode=mmap_mode)

        print(f'Loading labels from {self.label_path} ...')
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        self.label = np.asarray(self.label, dtype=np.int64)

        if len(self.data) != len(self.label):
            raise ValueError(
                f'Data and label size mismatch: '
                f'{len(self.data)} vs {len(self.label)}'
            )

        if self.debug:
            max_samples = min(100, len(self.label))
            self.data = self.data[:max_samples]
            self.label = self.label[:max_samples]
            self.sample_name = self.sample_name[:max_samples]

        print(
            f'Loaded {len(self.label)} samples, '
            f'data shape: {self.data.shape}'
        )

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.asarray(self.data[index], dtype=np.float32)
        label = int(self.label[index])

        # 当前 Penn Action 预处理阶段已经统一到固定长度，
        # 因此这里不再进行随机裁剪或额外插值。
        return data_numpy, label, index

    def top_k(self, score, top_k):
        """
        计算 Top-k 准确率。
        """
        score = np.asarray(score)

        if score.ndim != 2:
            raise ValueError(
                f'score must have shape (N, num_class), got {score.shape}'
            )

        if len(score) != len(self.label):
            raise ValueError(
                f'Score and label size mismatch: '
                f'{len(score)} vs {len(self.label)}'
            )

        rank = score.argsort(axis=1)
        hit = [
            int(label) in rank[i, -top_k:]
            for i, label in enumerate(self.label)
        ]

        return float(np.mean(hit))
