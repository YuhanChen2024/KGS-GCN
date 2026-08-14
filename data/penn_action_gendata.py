import os
import pickle

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


DATA_DIR = 'data/Penn_Action'
LABEL_DIR = os.path.join(DATA_DIR, 'labels')
OUT_DIR = os.path.join(DATA_DIR, 'processed_data')

MAX_FRAME = 100
NUM_JOINT = 13
NUM_PERSON = 1
NUM_CHANNEL = 3


ACTION_NAMES = [
    'baseball_pitch',
    'baseball_swing',
    'bench_press',
    'bowl',
    'clean_and_jerk',
    'golf_swing',
    'jump_rope',
    'jumping_jacks',
    'pullup',
    'pushup',
    'situp',
    'squat',
    'strum_guitar',
    'tennis_forehand',
    'tennis_serve',
]

ACTION_TO_LABEL = {
    name: index
    for index, name in enumerate(ACTION_NAMES)
}


def read_scalar(value):
    """
    将 scipy.io.loadmat 读取出的标量安全转换为 Python 标量。
    """
    value = np.asarray(value).squeeze()

    if value.size != 1:
        raise ValueError(f'Expected scalar value, got shape {value.shape}')

    return value.item()


def read_action_name(value):
    """
    兼容 MATLAB 字符串、numpy 字符串和 object 类型。
    """
    value = np.asarray(value).squeeze()

    if value.dtype.kind in {'U', 'S'}:
        return str(value)

    if value.dtype == object:
        return str(value.item())

    return str(value)


def normalize_coordinates(x, y):
    """
    将每个视频的二维关键点归一化到大致 [-1, 1] 范围。

    返回：
        x_norm, y_norm
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)

    width = max_x - min_x
    height = max_y - min_y
    scale = max(width, height) / 2.0

    if scale < 1e-6:
        scale = 1.0

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    x_norm = (x - center_x) / scale
    y_norm = (y - center_y) / scale

    return x_norm, y_norm


def process_one_mat(mat_path):
    mat = loadmat(mat_path)

    required_keys = [
        'x',
        'y',
        'visibility',
        'train',
        'action',
    ]

    missing_keys = [
        key for key in required_keys
        if key not in mat
    ]

    if missing_keys:
        raise KeyError(
            f'{mat_path} is missing keys: {missing_keys}'
        )

    x = np.asarray(mat['x'], dtype=np.float32)
    y = np.asarray(mat['y'], dtype=np.float32)
    visibility = np.asarray(mat['visibility'], dtype=np.float32)

    if x.ndim != 2 or y.ndim != 2 or visibility.ndim != 2:
        raise ValueError(
            f'Invalid shape in {mat_path}: '
            f'x={x.shape}, y={y.shape}, visibility={visibility.shape}'
        )

    if x.shape != y.shape or x.shape != visibility.shape:
        raise ValueError(
            f'Shape mismatch in {mat_path}: '
            f'x={x.shape}, y={y.shape}, visibility={visibility.shape}'
        )

    num_frame, num_joint = x.shape

    if num_joint != NUM_JOINT:
        raise ValueError(
            f'Expected {NUM_JOINT} joints, got {num_joint} '
            f'in {mat_path}'
        )

    action_name = read_action_name(mat['action'])

    if action_name not in ACTION_TO_LABEL:
        raise ValueError(
            f'Unknown action {action_name} in {mat_path}'
        )

    train_flag = int(read_scalar(mat['train']))

    x_norm, y_norm = normalize_coordinates(x, y)

    data = np.zeros(
        (NUM_CHANNEL, MAX_FRAME, NUM_JOINT, NUM_PERSON),
        dtype=np.float32
    )

    valid_frames = min(num_frame, MAX_FRAME)

    data[0, :valid_frames, :, 0] = x_norm[:valid_frames]
    data[1, :valid_frames, :, 0] = y_norm[:valid_frames]
    data[2, :valid_frames, :, 0] = visibility[:valid_frames]

    label = ACTION_TO_LABEL[action_name]

    return data, label, train_flag


def generate_dataset():
    if not os.path.isdir(LABEL_DIR):
        raise FileNotFoundError(
            f'Label directory does not exist: {LABEL_DIR}'
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    mat_files = sorted(
        file_name
        for file_name in os.listdir(LABEL_DIR)
        if file_name.endswith('.mat')
    )

    if not mat_files:
        raise RuntimeError(
            f'No .mat files found in {LABEL_DIR}'
        )

    train_data = []
    test_data = []
    train_label = []
    test_label = []
    train_names = []
    test_names = []

    print(f'Found {len(mat_files)} MATLAB files.')

    for file_name in tqdm(mat_files):
        mat_path = os.path.join(LABEL_DIR, file_name)

        data, label, train_flag = process_one_mat(mat_path)

        if train_flag == 1:
            train_data.append(data)
            train_label.append(label)
            train_names.append(file_name)
        else:
            test_data.append(data)
            test_label.append(label)
            test_names.append(file_name)

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)

    np.save(
        os.path.join(OUT_DIR, 'train_data_joint.npy'),
        train_data
    )

    with open(
        os.path.join(OUT_DIR, 'train_label.pkl'),
        'wb'
    ) as f:
        pickle.dump((train_names, train_label), f)

    np.save(
        os.path.join(OUT_DIR, 'test_data_joint.npy'),
        test_data
    )

    with open(
        os.path.join(OUT_DIR, 'test_label.pkl'),
        'wb'
    ) as f:
        pickle.dump((test_names, test_label), f)

    print('Penn Action preprocessing finished.')
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    print(f'Train samples: {len(train_label)}')
    print(f'Test samples: {len(test_label)}')
    print(f'Output directory: {OUT_DIR}')


if __name__ == '__main__':
    generate_dataset()
