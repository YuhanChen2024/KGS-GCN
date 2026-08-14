import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def import_class(name):
    """
    动态导入图结构类。

    示例：
        graph.penn_action.Graph
    """
    components = name.split('.')
    mod = __import__(components[0])

    for component in components[1:]:
        mod = getattr(mod, component)

    return mod


def conv_init(conv):
    """
    卷积层 Kaiming 初始化。
    """
    if conv.weight is not None:
        nn.init.kaiming_normal_(
            conv.weight,
            mode='fan_out'
        )

    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    """
    BatchNorm 初始化。
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class SkeletonNorm(nn.Module):
    """
    Skeleton normalization.

    对每一帧的人体关键点进行中心化，并按最大关节距离缩放，
    从而将骨架映射到近似 [-0.8, 0.8] 的坐标范围。

    该模块用于消除 Penn Action 原始像素坐标尺度差异，
    使 Gaussian Splatting 的坐标空间保持稳定。
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (N, C, T, V)

        Returns:
            x_norm: 归一化骨架，shape (N, C, T, V)
            mean: 每帧中心，shape (N, C, T, 1)
            scale: 每帧缩放尺度，shape (N, 1, T, 1)
        """
        _, _, _, _ = x.size()

        mean = x.mean(dim=3, keepdim=True)
        x_centered = x - mean

        distance = x_centered.norm(dim=1)
        max_distance = distance.max(
            dim=2,
            keepdim=True
        )[0]

        scale = 0.8 / (max_distance + 1e-6)
        scale = scale.unsqueeze(1)

        x_norm = x_centered * scale

        return x_norm, mean, scale


class KinematicGaussianSplatting(nn.Module):
    """
    Kinematics-Driven Gaussian Splatting.

    对每个关节构建一个二维各向异性 Gaussian：
    1. Gaussian 中心由关节位置决定；
    2. Gaussian 方向由关节速度方向决定；
    3. Gaussian 长轴尺度由速度大小决定；
    4. 最终将所有关节 Gaussian 聚合为时序热图。

    对于输入 C=3 的情况，依次渲染：
        XY、YZ、ZX 三个正交投影视图。

    对于输入 C=2 的情况，仅渲染：
        XY 视图。
    """

    def __init__(
        self,
        num_point,
        in_channels=3,
        img_size=32
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_point = num_point

        self.log_scale = nn.Parameter(
            torch.zeros(num_point, 1) - 2.0
        )

        self.kinematic_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                in_channels + 1,
                kernel_size=1
            )
        )

        self.register_buffer(
            'grid',
            self._create_grid(img_size)
        )

    @staticmethod
    def _create_grid(size):
        """
        创建 [-1, 1] x [-1, 1] 的二维坐标网格。

        Returns:
            grid: shape (1, 1, 1, H, W, 2)
        """
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)

        grid_y, grid_x = torch.meshgrid(
            y,
            x,
            indexing='ij'
        )

        return torch.stack(
            [grid_x, grid_y],
            dim=-1
        ).view(1, 1, 1, size, size, 2)

    @staticmethod
    def compute_covariance(
        scale_base,
        velocity_norm,
        velocity_direction
    ):
        """
        根据速度大小和速度方向构建二维 Gaussian 协方差矩阵。

        Sigma = R * S * S^T * R^T

        Args:
            scale_base:
                基础尺度，shape (N*M, T, V)

            velocity_norm:
                速度模长，shape (N*M, T, V)

            velocity_direction:
                速度单位方向，shape (N*M, T, V, 2)

        Returns:
            cov_xx, cov_yy, cov_xy:
                协方差矩阵的三个独立元素，
                每个 shape 均为 (N*M, T, V)。
        """
        scale_x = scale_base * (
            1.0 + 2.0 * torch.tanh(velocity_norm)
        )

        scale_y = scale_base

        cos_theta = velocity_direction[..., 0]
        sin_theta = velocity_direction[..., 1]

        scale_x_squared = scale_x ** 2
        scale_y_squared = scale_y ** 2

        cov_xx = (
            scale_x_squared * cos_theta ** 2
            + scale_y_squared * sin_theta ** 2
        )

        cov_yy = (
            scale_x_squared * sin_theta ** 2
            + scale_y_squared * cos_theta ** 2
        )

        cov_xy = (
            (scale_x_squared - scale_y_squared)
            * sin_theta
            * cos_theta
        )

        return cov_xx, cov_yy, cov_xy

    def forward(self, x, velocity):
        """
        Args:
            x:
                归一化骨架，
                shape (N*M, C, T, V)。

            velocity:
                归一化速度，
                shape (N*M, C, T, V)。

        Returns:
            gaussian_maps:
                shape (N*M, Views, T, H, W)。

            mu_list:
                每个视图的关节均值，
                每个元素 shape 为 (N*M, 2, T, V)。

            sigma_list:
                每个视图的协方差矩阵，
                每个元素 shape 为 (N*M, T, V, 2, 2)。
        """
        _, channels, _, num_joint = x.size()

        kinematics = torch.cat(
            [x, velocity],
            dim=1
        )

        dynamics = self.kinematic_net(kinematics)

        delta_scale = torch.sigmoid(
            dynamics[:, -1:, :, :]
        )

        base_scale = (
            torch.exp(self.log_scale).view(
                1,
                1,
                1,
                num_joint
            )
            * (0.5 + delta_scale)
        )

        if channels == 3:
            projection_pairs = [
                (0, 1),
                (1, 2),
                (2, 0)
            ]
        else:
            projection_pairs = [
                (0, 1)
            ]

        maps_list = []
        mu_list = []
        sigma_list = []

        for idx_u, idx_v in projection_pairs:
            position_view = x[
                :,
                [idx_u, idx_v],
                :,
                :
            ]

            velocity_view = velocity[
                :,
                [idx_u, idx_v],
                :,
                :
            ]

            velocity_norm = torch.norm(
                velocity_view,
                dim=1,
                keepdim=True
            )

            velocity_direction = velocity_view / (
                velocity_norm + 1e-6
            )

            cov_xx, cov_yy, cov_xy = self.compute_covariance(
                base_scale.squeeze(1),
                velocity_norm.squeeze(1),
                velocity_direction.permute(0, 2, 3, 1)
            )

            mu = position_view.permute(
                0,
                2,
                3,
                1
            ).unsqueeze(-2).unsqueeze(-2)

            diff = self.grid - mu

            dx = diff[..., 0]
            dy = diff[..., 1]

            determinant = (
                cov_xx * cov_yy
                - cov_xy ** 2
                + 1e-6
            )

            determinant = determinant.unsqueeze(-1).unsqueeze(-1)

            cov_xx_expanded = cov_xx.unsqueeze(-1).unsqueeze(-1)
            cov_yy_expanded = cov_yy.unsqueeze(-1).unsqueeze(-1)
            cov_xy_expanded = cov_xy.unsqueeze(-1).unsqueeze(-1)

            inv_a = cov_yy_expanded / determinant
            inv_b = -cov_xy_expanded / determinant
            inv_c = cov_xx_expanded / determinant

            mahalanobis_distance = (
                inv_a * dx ** 2
                + 2 * inv_b * dx * dy
                + inv_c * dy ** 2
            )

            gaussian = torch.exp(
                -0.5 * mahalanobis_distance
            )

            heatmap = torch.sum(
                gaussian,
                dim=2
            )

            heatmap_max = heatmap.max(
                dim=-1,
                keepdim=True
            )[0].max(
                dim=-2,
                keepdim=True
            )[0]

            heatmap = heatmap / (
                heatmap_max + 1e-6
            )

            covariance_matrix = torch.stack(
                [
                    torch.stack(
                        [cov_xx, cov_xy],
                        dim=-1
                    ),
                    torch.stack(
                        [cov_xy, cov_yy],
                        dim=-1
                    )
                ],
                dim=-1
            )

            maps_list.append(heatmap)
            mu_list.append(position_view)
            sigma_list.append(covariance_matrix)

        gaussian_maps = torch.stack(
            maps_list,
            dim=1
        )

        return gaussian_maps, mu_list, sigma_list


class ProbabilisticTopology(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, mu_list, sigma_list):
        """
        Args:
            mu_list:
                每个视图对应的关节位置列表。
                元素 shape: (N*M, 2, T, V)。

            sigma_list:
                每个视图对应的协方差列表。
                元素 shape: (N*M, T, V, 2, 2)。

        Returns:
            A_prior:
                shape (N*M, V, V)。
        """
        total_adjacency = 0.0

        for mu, sigma in zip(mu_list, sigma_list):
            mu = mu[:, :, ::4, :].permute(
                0,
                2,
                3,
                1
            )

            sigma = sigma[:, ::4, :, :, :]

            mu_i = mu.unsqueeze(3)
            mu_j = mu.unsqueeze(2)

            sigma_i = sigma.unsqueeze(3)
            sigma_j = sigma.unsqueeze(2)

            sigma_avg = 0.5 * (
                sigma_i + sigma_j
            )

            sa_00 = sigma_avg[..., 0, 0]
            sa_01 = sigma_avg[..., 0, 1]
            sa_10 = sigma_avg[..., 1, 0]
            sa_11 = sigma_avg[..., 1, 1]

            det_avg = (
                sa_00 * sa_11
                - sa_01 * sa_10
                + 1e-8
            )

            dx = mu_i[..., 0] - mu_j[..., 0]
            dy = mu_i[..., 1] - mu_j[..., 1]

            distance_term = 0.125 * (
                (
                    dx ** 2 * sa_11
                    + dy ** 2 * sa_00
                    - 2 * dx * dy * sa_01
                )
                / det_avg
            )

            det_i = (
                sigma_i[..., 0, 0]
                * sigma_i[..., 1, 1]
                - sigma_i[..., 0, 1]
                * sigma_i[..., 1, 0]
                + 1e-8
            )

            det_j = (
                sigma_j[..., 0, 0]
                * sigma_j[..., 1, 1]
                - sigma_j[..., 0, 1]
                * sigma_j[..., 1, 0]
                + 1e-8
            )

            covariance_term = 0.5 * torch.log(
                det_avg / torch.sqrt(det_i * det_j)
            )

            bhattacharyya_distance = (
                distance_term + covariance_term
            )

            adjacency = torch.exp(
                -bhattacharyya_distance
            )

            total_adjacency = (
                total_adjacency
                + adjacency.mean(dim=1)
            )

        return total_adjacency / len(mu_list)


class TemporalConv(nn.Module):
    """
    基础时序卷积模块。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1
    ):
        super().__init__()

        padding = (
            kernel_size
            + (kernel_size - 1) * (dilation - 1)
            - 1
        ) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1),
            dilation=(dilation, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class MultiScaleTemporalConv(nn.Module):
    """
    多尺度时序卷积模块。

    使用多个不同 dilation 的 temporal convolution 分支，
    同时提取短时和长时动作模式。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=(1, 2, 3, 4),
        residual=True,
        residual_kernel_size=1
    ):
        super().__init__()

        num_branches = len(dilations) + 2

        assert out_channels % num_branches == 0, (
            'out_channels must be divisible by '
            'the number of temporal branches.'
        )

        branch_channels = out_channels // num_branches

        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
            kernel_sizes = kernel_size
        else:
            kernel_sizes = [
                kernel_size
                for _ in dilations
            ]

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=1
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation
                    )
                )
                for ks, dilation in zip(
                    kernel_sizes,
                    dilations
                )
            ]
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=(3, 1),
                    stride=(stride, 1),
                    padding=(1, 0)
                ),
                nn.BatchNorm2d(branch_channels)
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    stride=(stride, 1)
                ),
                nn.BatchNorm2d(branch_channels)
            )
        )

        if not residual:
            self.residual = lambda x: 0

        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x

        else:
            self.residual = TemporalConv(
                in_channels,
                out_channels,
                kernel_size=residual_kernel_size,
                stride=stride
            )

    def forward(self, x):
        residual = self.residual(x)

        branch_outputs = [
            branch(x)
            for branch in self.branches
        ]

        output = torch.cat(
            branch_outputs,
            dim=1
        )

        output = output + residual

        return output


class CTRGCVisual(nn.Module):
    """
    带概率拓扑先验注入的 Channel-wise Topology Refinement GCN。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        rel_reduction=8
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels in (3, 9):
            rel_channels = 8
        else:
            rel_channels = in_channels // rel_reduction

        self.conv1 = nn.Conv2d(
            in_channels,
            rel_channels,
            kernel_size=1
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            rel_channels,
            kernel_size=1
        )

        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.conv4 = nn.Conv2d(
            rel_channels,
            out_channels,
            kernel_size=1
        )

        self.tanh = nn.Tanh()

        # 用于控制概率拓扑先验 A_prior 的注入强度。
        self.beta = nn.Parameter(
            torch.zeros(1)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)

            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)

    def forward(
        self,
        x,
        A=None,
        A_prior=None,
        alpha=1
    ):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)

        relation = self.tanh(
            x1.unsqueeze(-1)
            - x2.unsqueeze(-2)
        )

        ctr_adjacency = self.conv4(
            relation
        ) * alpha

        if A is not None:
            ctr_adjacency = (
                ctr_adjacency
                + A.unsqueeze(0).unsqueeze(0)
            )

        if A_prior is not None:
            ctr_adjacency = (
                ctr_adjacency
                + self.beta * A_prior.unsqueeze(1)
            )

        output = torch.einsum(
            'ncuv,nctv->nctu',
            ctr_adjacency,
            x3
        )

        return output, ctr_adjacency


class UnitGCNVisual(nn.Module):
    """
    图卷积单元，包含：

    1. CTR-GCN 自适应关系学习；
    2. Gaussian topology prior 注入；
    3. Visual Context Gating。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        adaptive=True,
        residual=True
    ):
        super().__init__()

        self.adaptive = adaptive
        self.num_subset = A.shape[0]

        self.convs = nn.ModuleList(
            [
                CTRGCVisual(
                    in_channels,
                    out_channels
                )
                for _ in range(self.num_subset)
            ]
        )

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if adaptive:
            self.PA = nn.Parameter(
                torch.from_numpy(
                    A.astype(np.float32)
                )
            )
        else:
            self.A = Variable(
                torch.from_numpy(
                    A.astype(np.float32)
                ),
                requires_grad=False
            )

        self.alpha = nn.Parameter(
            torch.zeros(1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.vis_project = nn.Sequential(
            nn.Conv2d(
                128,
                out_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)

        bn_init(self.bn, 1e-6)

    def forward(
        self,
        x,
        A_prior=None,
        visual_feat=None
    ):
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.to(x.device)

        output = None
        learned_topologies = []

        for subset_index in range(self.num_subset):
            z, topology = self.convs[subset_index](
                x,
                A[subset_index],
                A_prior,
                self.alpha
            )

            output = z if output is None else output + z
            learned_topologies.append(topology)

        output = self.bn(output)
        output = output + self.down(x)

        if visual_feat is not None:
            _, _, temporal_length, num_joint = output.size()

            visual_feature = F.interpolate(
                visual_feat,
                size=temporal_length,
                mode='linear',
                align_corners=False
            )

            visual_feature = visual_feature.unsqueeze(
                -1
            ).expand(
                -1,
                -1,
                -1,
                num_joint
            )

            gate = self.vis_project(visual_feature)

            output = output * (1.0 + gate)

        output = self.relu(output)

        # 保持原逻辑：每层仅返回第一个 subset 的 topology，
        # 用于 topology consistency loss。
        return output, learned_topologies[0]


class UnitTCN(nn.Module):
    """
    单尺度时序卷积单元。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(
            self.conv(x)
        )


class TCN_GCN_Unit_Visual(nn.Module):
    """
    GCN + Multi-scale TCN 组合单元。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        adaptive=True,
        kernel_size=5,
        dilations=(1, 2)
    ):
        super().__init__()

        self.gcn1 = UnitGCNVisual(
            in_channels,
            out_channels,
            A,
            adaptive=adaptive
        )

        self.tcn1 = MultiScaleTemporalConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False
        )

        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x

        else:
            self.residual = UnitTCN(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride
            )

    def forward(
        self,
        x,
        A_prior=None,
        visual_feat=None
    ):
        gcn_feature, learned_topology = self.gcn1(
            x,
            A_prior,
            visual_feat
        )

        output = self.relu(
            self.tcn1(gcn_feature)
            + self.residual(x)
        )

        return output, learned_topology


class VisualBranch(nn.Module):
    """
    对 Gaussian heatmaps 提取视觉时序特征。

    输入：
        (N*M, Views, T, H, W)

    输出：
        (N*M, 128, T)
    """

    def __init__(
        self,
        in_views=3,
        base_channels=32
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_views,
                base_channels,
                kernel_size=3,
                padding=1,
                stride=2
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                stride=2
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.project = nn.Linear(
            base_channels * 4,
            128
        )

    def forward(self, x):
        """
        Args:
            x:
                Gaussian heatmaps，
                shape (N*M, Views, T, H, W)。

        Returns:
            feature:
                shape (N*M, 128, T)。
        """
        batch_size, num_views, temporal_length, height, width = x.size()

        x = x.permute(
            0,
            2,
            1,
            3,
            4
        ).reshape(
            batch_size * temporal_length,
            num_views,
            height,
            width
        )

        feature = self.encoder(x)

        feature = feature.view(
            batch_size,
            temporal_length,
            -1
        )

        feature = self.project(feature)

        feature = feature.permute(
            0,
            2,
            1
        )

        return feature


class Model(nn.Module):
    """
    Kinematics-Driven Gaussian Splatting GCN.

    核心流程：

        Skeleton Input
            ↓
        Data BatchNorm
            ↓
        SkeletonNorm + Velocity
            ↓
        Kinematic Gaussian Splatting
            ↓
        Probabilistic Topology Prior
            ↓
        Visual Branch
            ↓
        CTR-GCN + Multi-scale TCN Backbone
            ↓
        GCN Feature + Visual Feature Fusion
            ↓
        Classification
    """

    def __init__(
        self,
        num_class=15,
        num_point=13,
        num_person=1,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,
        adaptive=True
    ):
        super().__init__()

        if graph is None:
            raise ValueError(
                'graph must be specified, e.g. '
                '"graph.penn_action.Graph".'
            )

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels

        self.data_bn = nn.BatchNorm1d(
            num_person * in_channels * num_point
        )

        self.skeleton_norm = SkeletonNorm(
            in_channels=in_channels
        )

        self.splatting_net = KinematicGaussianSplatting(
            num_point=num_point,
            in_channels=in_channels,
            img_size=32
        )

        self.topology_net = ProbabilisticTopology()

        # 保持原始逻辑：
        # 若 C=3，Gaussian Splatting 生成 XY/YZ/ZX 三个视图；
        # 否则仅生成一个 XY 视图。
        num_views = 3 if in_channels == 3 else 1

        self.visual_net = VisualBranch(
            in_views=num_views,
            base_channels=32
        )

        base_channel = 64

        self.l1 = TCN_GCN_Unit_Visual(
            in_channels,
            base_channel,
            A,
            residual=False,
            adaptive=adaptive
        )

        self.l2 = TCN_GCN_Unit_Visual(
            base_channel,
            base_channel,
            A,
            adaptive=adaptive
        )

        self.l3 = TCN_GCN_Unit_Visual(
            base_channel,
            base_channel,
            A,
            adaptive=adaptive
        )

        self.l4 = TCN_GCN_Unit_Visual(
            base_channel,
            base_channel,
            A,
            adaptive=adaptive
        )

        self.l5 = TCN_GCN_Unit_Visual(
            base_channel,
            base_channel * 2,
            A,
            stride=2,
            adaptive=adaptive
        )

        self.l6 = TCN_GCN_Unit_Visual(
            base_channel * 2,
            base_channel * 2,
            A,
            adaptive=adaptive
        )

        self.l7 = TCN_GCN_Unit_Visual(
            base_channel * 2,
            base_channel * 2,
            A,
            adaptive=adaptive
        )

        self.l8 = TCN_GCN_Unit_Visual(
            base_channel * 2,
            base_channel * 4,
            A,
            stride=2,
            adaptive=adaptive
        )

        self.l9 = TCN_GCN_Unit_Visual(
            base_channel * 4,
            base_channel * 4,
            A,
            adaptive=adaptive
        )

        self.l10 = TCN_GCN_Unit_Visual(
            base_channel * 4,
            base_channel * 4,
            A,
            adaptive=adaptive
        )

        self.fc = nn.Linear(
            base_channel * 4 + 128,
            num_class
        )

        nn.init.normal_(
            self.fc.weight,
            0,
            math.sqrt(2.0 / num_class)
        )

        bn_init(self.data_bn, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        """
        Args:
            x:
                输入骨架序列，
                shape (N, C, T, V, M)。

        Returns:
            output:
                分类 logits，
                shape (N, num_class)。

            learned_topologies:
                10 个 GCN block 学到的 topology 列表。

            A_prior:
                Gaussian/Bhattacharyya 生成的拓扑先验，
                shape (N*M, V, V)。
        """
        batch_size, channels, temporal_length, num_joint, num_person = x.size()

        # ---------------------------------------------------------
        # 1. 输入骨架标准化
        # ---------------------------------------------------------
        x = x.permute(
            0,
            4,
            3,
            1,
            2
        ).contiguous().view(
            batch_size,
            num_person * num_joint * channels,
            temporal_length
        )

        x = self.data_bn(x)

        x = x.view(
            batch_size,
            num_person,
            num_joint,
            channels,
            temporal_length
        ).permute(
            0,
            1,
            3,
            4,
            2
        ).contiguous().view(
            batch_size * num_person,
            channels,
            temporal_length,
            num_joint
        )

        # ---------------------------------------------------------
        # 2. 速度计算
        # ---------------------------------------------------------
        velocity_raw = (
            x[:, :, 1:, :]
            - x[:, :, :-1, :]
        )

        velocity_raw = torch.cat(
            [
                velocity_raw[:, :, 0:1, :],
                velocity_raw
            ],
            dim=2
        )

        # ---------------------------------------------------------
        # 3. 骨架与速度归一化
        # ---------------------------------------------------------
        x_norm, _, _ = self.skeleton_norm(x)

        velocity_norm, _, _ = self.skeleton_norm(
            velocity_raw
        )

        # ---------------------------------------------------------
        # 4. Kinematic Gaussian Splatting
        # ---------------------------------------------------------
        gaussian_maps, mu_list, sigma_list = self.splatting_net(
            x_norm,
            velocity_norm
        )

        # ---------------------------------------------------------
        # 5. Bhattacharyya 概率拓扑先验
        # ---------------------------------------------------------
        A_prior = self.topology_net(
            mu_list,
            sigma_list
        )

        # ---------------------------------------------------------
        # 6. Gaussian visual feature extraction
        # ---------------------------------------------------------
        visual_feature = self.visual_net(
            gaussian_maps
        )

        # ---------------------------------------------------------
        # 7. CTR-GCN + Multi-scale TCN Backbone
        # ---------------------------------------------------------
        x1, topology_1 = self.l1(
            x,
            A_prior
        )

        x2, topology_2 = self.l2(
            x1,
            A_prior
        )

        x3, topology_3 = self.l3(
            x2,
            A_prior
        )

        x4, topology_4 = self.l4(
            x3,
            A_prior,
            visual_feature
        )

        x5, topology_5 = self.l5(
            x4,
            A_prior
        )

        x6, topology_6 = self.l6(
            x5,
            A_prior
        )

        x7, topology_7 = self.l7(
            x6,
            A_prior,
            visual_feature
        )

        x8, topology_8 = self.l8(
            x7,
            A_prior
        )

        x9, topology_9 = self.l9(
            x8,
            A_prior
        )

        x10, topology_10 = self.l10(
            x9,
            A_prior
        )

        # ---------------------------------------------------------
        # 8. GCN Feature Pooling
        # ---------------------------------------------------------
        gcn_feature = x10.view(
            batch_size,
            num_person,
            -1,
            temporal_length // 4,
            num_joint
        ).mean(
            dim=4
        ).mean(
            dim=3
        ).mean(
            dim=1
        )

        # ---------------------------------------------------------
        # 9. Visual Feature Pooling
        # ---------------------------------------------------------
        visual_feature_pool = visual_feature.view(
            batch_size,
            num_person,
            -1,
            temporal_length
        ).mean(
            dim=3
        ).mean(
            dim=1
        )

        # ---------------------------------------------------------
        # 10. Feature Fusion and Classification
        # ---------------------------------------------------------
        final_feature = torch.cat(
            [
                gcn_feature,
                visual_feature_pool
            ],
            dim=1
        )

        final_feature = self.drop_out(
            final_feature
        )

        output = self.fc(
            final_feature
        )

        learned_topologies = [
            topology_1,
            topology_2,
            topology_3,
            topology_4,
            topology_5,
            topology_6,
            topology_7,
            topology_8,
            topology_9,
            topology_10
        ]

        return output, learned_topologies, A_prior
