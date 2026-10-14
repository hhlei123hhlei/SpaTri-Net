import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. SPIN
# Spatial Prior Injection Network
# =========================================================
class SPIN(nn.Module):
    """
    Revised SPIN:
    Template-space Coordinate Prior Injection.

    原始实现:
        根据 crop / resize 后的当前输入尺寸
        重新生成 [-1,1] 坐标。

    问题:
        坐标是 crop-relative coordinates。

    修订实现:
        Dataset 在完整注册模板空间中生成坐标，
        并让坐标与 MRI 经历完全相同的 crop / resize。

        SPIN 不再负责创建坐标，
        只负责将 anatomical coordinates 注入 MRI。

    输入:
        x:
            [B, 1, D, H, W]

        coords:
            [B, 3, D, H, W]

    输出:
        [B, 4, D, H, W]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        coords,
    ):

        # =====================================================
        # Shape checks
        # =====================================================
        if x.ndim != 5:

            raise ValueError(
                f"x 应为 "
                f"[B,C,D,H,W], "
                f"实际为 "
                f"{tuple(x.shape)}"
            )

        if coords.ndim != 5:

            raise ValueError(
                f"coords 应为 "
                f"[B,3,D,H,W], "
                f"实际为 "
                f"{tuple(coords.shape)}"
            )

        # =====================================================
        # MRI channel
        # =====================================================
        if x.size(1) != 1:

            raise ValueError(
                f"SpaTri-Net 当前只使用"
                f"单模态 3D sMRI。\n"
                f"Expected MRI channels = 1\n"
                f"Actual = {x.size(1)}"
            )

        # =====================================================
        # Coordinate channels
        # =====================================================
        if coords.size(1) != 3:

            raise ValueError(
                f"SPIN 需要三个坐标通道 "
                f"(D/H/W)，"
                f"实际为 {coords.size(1)}"
            )

        # =====================================================
        # Batch
        # =====================================================
        if (
            x.size(0)
            != coords.size(0)
        ):

            raise ValueError(
                f"MRI / coordinates "
                f"batch size 不一致: "
                f"{x.size(0)} vs "
                f"{coords.size(0)}"
            )

        # =====================================================
        # Spatial dimensions
        # =====================================================
        if (
            tuple(
                x.shape[2:]
            )
            != tuple(
                coords.shape[2:]
            )
        ):

            raise ValueError(
                f"MRI / coordinates "
                f"空间尺寸不一致:\n"
                f"MRI: "
                f"{tuple(x.shape)}\n"
                f"Coords: "
                f"{tuple(coords.shape)}"
            )

        # =====================================================
        # AMP / CUDA compatibility
        # =====================================================
        coords = coords.to(
            device=x.device,
            dtype=x.dtype,
            non_blocking=True,
        )

        # =====================================================
        # SPIN injection
        # =====================================================
        return torch.cat(
            [
                x,
                coords,
            ],
            dim=1,
        )


# =========================================================
# 2. Core sub-module of AC-STA
# Adaptive Cross-Plane Spatial Triplet Attention
# =========================================================
class AC_SpatialAttentionSubModule(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        reduction_ratio=2,
        kernel_size=3,
    ):

        super().__init__()

        reduction_channels = max(
            in_channels
            // reduction_ratio,
            1,
        )

        # =====================================================
        # Q projection
        # =====================================================
        self.conv_Q = nn.Sequential(

            nn.Conv2d(
                in_channels,
                reduction_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                reduction_channels
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True,
            ),
        )

        # =====================================================
        # K projection
        # =====================================================
        self.conv_K = nn.Conv2d(
            in_channels,
            reduction_channels,
            kernel_size=1,
            bias=False,
        )

        # =====================================================
        # V projection
        # =====================================================
        self.conv_V = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=False,
        )

        # =====================================================
        # Depthwise branches
        # =====================================================
        self.depthwise_avg = (
            nn.Conv2d(
                reduction_channels,
                reduction_channels,
                kernel_size=kernel_size,
                padding=(
                    kernel_size // 2
                ),
                groups=(
                    reduction_channels
                ),
                bias=False,
            )
        )

        self.depthwise_max = (
            nn.Conv2d(
                reduction_channels,
                reduction_channels,
                kernel_size=kernel_size,
                padding=(
                    kernel_size // 2
                ),
                groups=(
                    reduction_channels
                ),
                bias=False,
            )
        )

        # =====================================================
        # Adaptive Avg/Max gate
        # =====================================================
        self.gate = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                1
            ),

            nn.Conv2d(
                reduction_channels,
                2,
                kernel_size=1,
                bias=True,
            ),

            nn.Softmax(
                dim=1
            ),
        )

        # =====================================================
        # Spatial modulation map
        # =====================================================
        self.conv_A = nn.Sequential(

            nn.Conv2d(
                reduction_channels,
                1,
                kernel_size=1,
                bias=False,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        x,
    ):

        # =====================================================
        # Q/K/V
        # =====================================================
        Q = self.conv_Q(
            x
        )

        K = self.conv_K(
            x
        )

        V = self.conv_V(
            x
        )

        # =====================================================
        # Multi-context K
        # =====================================================
        K_avg = (
            self.depthwise_avg(
                K
            )
        )

        K_max = (
            self.depthwise_max(
                K
            )
        )

        # =====================================================
        # Element-wise Q-K interaction
        # =====================================================
        A_avg = (
            Q * K_avg
        )

        A_max = (
            Q * K_max
        )

        # =====================================================
        # Adaptive mixture
        # =====================================================
        gate_weights = (
            self.gate(
                A_avg
                + A_max
            )
        )

        w_avg = (
            gate_weights[
                :, 0:1
            ]
        )

        w_max = (
            gate_weights[
                :, 1:2
            ]
        )

        A_combined = (
            w_avg
            * A_avg
            +
            w_max
            * A_max
        )

        # =====================================================
        # Spatial modulation mask
        # =====================================================
        A_final = (
            self.conv_A(
                A_combined
            )
        )

        # =====================================================
        # Local residual modulation
        # =====================================================
        return (
            x
            + V
            * A_final
        )


# =========================================================
# 3. AC-STA
# Adaptive Cross-Plane Spatial Triplet Attention
# =========================================================
class AC_STA(
    nn.Module
):

    def __init__(
        self,
        channels,
        kernel_size=3,
    ):

        super().__init__()

        self.hw_module = (
            AC_SpatialAttentionSubModule(
                in_channels=channels,
                kernel_size=kernel_size,
            )
        )

        self.cw_module = (
            AC_SpatialAttentionSubModule(
                in_channels=channels,
                kernel_size=kernel_size,
            )
        )

        self.hc_module = (
            AC_SpatialAttentionSubModule(
                in_channels=channels,
                kernel_size=kernel_size,
            )
        )

    def forward(
        self,
        x,
    ):

        b, c, d, h, w = (
            x.shape
        )

        # =====================================================
        # HW Plane
        #
        # [B,C,D,H,W]
        # ->
        # [B*D,C,H,W]
        # =====================================================
        x_hw = (
            x.permute(
                0,
                2,
                1,
                3,
                4,
            )
            .contiguous()
            .view(
                b * d,
                c,
                h,
                w,
            )
        )

        out_hw = (
            self.hw_module(
                x_hw
            )
            .view(
                b,
                d,
                c,
                h,
                w,
            )
            .permute(
                0,
                2,
                1,
                3,
                4,
            )
            .contiguous()
        )

        # =====================================================
        # DW Plane
        #
        # [B,C,D,H,W]
        # ->
        # [B*H,C,D,W]
        # =====================================================
        x_dw = (
            x.permute(
                0,
                3,
                1,
                2,
                4,
            )
            .contiguous()
            .view(
                b * h,
                c,
                d,
                w,
            )
        )

        out_dw = (
            self.cw_module(
                x_dw
            )
            .view(
                b,
                h,
                c,
                d,
                w,
            )
            .permute(
                0,
                2,
                3,
                1,
                4,
            )
            .contiguous()
        )

        # =====================================================
        # DH Plane
        #
        # [B,C,D,H,W]
        # ->
        # [B*W,C,D,H]
        # =====================================================
        x_dh = (
            x.permute(
                0,
                4,
                1,
                2,
                3,
            )
            .contiguous()
            .view(
                b * w,
                c,
                d,
                h,
            )
        )

        out_dh = (
            self.hc_module(
                x_dh
            )
            .view(
                b,
                w,
                c,
                d,
                h,
            )
            .permute(
                0,
                2,
                3,
                4,
                1,
            )
            .contiguous()
        )

        # =====================================================
        # Three-plane fusion
        # =====================================================
        return (
            out_hw
            + out_dw
            + out_dh
        ) / 3.0


# =========================================================
# 4. Residual Bottleneck 3D
# =========================================================
class ResidualBottleneck3D(
    nn.Module
):

    expansion = 4

    def __init__(
        self,
        in_ch,
        mid_ch,
        stride=1,
    ):

        super().__init__()

        out_ch = (
            mid_ch
            * self.expansion
        )

        # =====================================================
        # 1x1x1
        # =====================================================
        self.conv1 = nn.Conv3d(
            in_ch,
            mid_ch,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(
            mid_ch
        )

        # =====================================================
        # 3x3x3
        # =====================================================
        self.conv2 = nn.Conv3d(
            mid_ch,
            mid_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm3d(
            mid_ch
        )

        # =====================================================
        # 1x1x1
        # =====================================================
        self.conv3 = nn.Conv3d(
            mid_ch,
            out_ch,
            kernel_size=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm3d(
            out_ch
        )

        self.relu = nn.LeakyReLU(
            0.1,
            inplace=True,
        )

        # =====================================================
        # Shortcut
        # =====================================================
        self.shortcut = (
            nn.Sequential()
        )

        if (
            stride != 1
            or in_ch != out_ch
        ):

            self.shortcut = (
                nn.Sequential(

                    nn.Conv3d(
                        in_ch,
                        out_ch,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),

                    nn.BatchNorm3d(
                        out_ch
                    ),
                )
            )

    def forward(
        self,
        x,
    ):

        identity = (
            self.shortcut(
                x
            )
        )

        out = (
            self.conv1(
                x
            )
        )

        out = (
            self.bn1(
                out
            )
        )

        out = (
            self.relu(
                out
            )
        )

        out = (
            self.conv2(
                out
            )
        )

        out = (
            self.bn2(
                out
            )
        )

        out = (
            self.relu(
                out
            )
        )

        out = (
            self.conv3(
                out
            )
        )

        out = (
            self.bn3(
                out
            )
        )

        out = (
            out
            + identity
        )

        return (
            self.relu(
                out
            )
        )


# =========================================================
# 5. MS-SCE
# Multi-scale Spatial-Channel Explorer
# =========================================================
class MS_SCE(
    nn.Module
):

    def __init__(
        self,
        in_ch,
        out_ch,
        r=4,
    ):

        super().__init__()

        self.out_ch = out_ch

        # =====================================================
        # Scale 1
        # =====================================================
        self.branch1 = nn.Sequential(

            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_ch
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True,
            ),
        )

        # =====================================================
        # Scale 2
        # dilation = 2
        # =====================================================
        self.branch2 = nn.Sequential(

            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(
                    3,
                    1,
                    1,
                ),
                padding=(
                    2,
                    0,
                    0,
                ),
                dilation=(
                    2,
                    1,
                    1,
                ),
                bias=False,
            ),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(
                    1,
                    3,
                    1,
                ),
                padding=(
                    0,
                    2,
                    0,
                ),
                dilation=(
                    1,
                    2,
                    1,
                ),
                bias=False,
            ),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(
                    1,
                    1,
                    3,
                ),
                padding=(
                    0,
                    0,
                    2,
                ),
                dilation=(
                    1,
                    1,
                    2,
                ),
                bias=False,
            ),

            nn.BatchNorm3d(
                out_ch
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True,
            ),
        )

        # =====================================================
        # Scale 3
        # dilation = 3
        # =====================================================
        self.branch3 = nn.Sequential(

            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(
                    3,
                    1,
                    1,
                ),
                padding=(
                    3,
                    0,
                    0,
                ),
                dilation=(
                    3,
                    1,
                    1,
                ),
                bias=False,
            ),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(
                    1,
                    3,
                    1,
                ),
                padding=(
                    0,
                    3,
                    0,
                ),
                dilation=(
                    1,
                    3,
                    1,
                ),
                bias=False,
            ),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(
                    1,
                    1,
                    3,
                ),
                padding=(
                    0,
                    0,
                    3,
                ),
                dilation=(
                    1,
                    1,
                    3,
                ),
                bias=False,
            ),

            nn.BatchNorm3d(
                out_ch
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True,
            ),
        )

        # =====================================================
        # Channel competition
        # =====================================================
        self.gap = (
            nn.AdaptiveAvgPool3d(
                1
            )
        )

        self.gmp = (
            nn.AdaptiveMaxPool3d(
                1
            )
        )

        mid_ch = max(
            out_ch // r,
            16,
        )

        self.fc_compact = (
            nn.Sequential(

                nn.Conv3d(
                    out_ch,
                    mid_ch,
                    kernel_size=1,
                    bias=False,
                ),

                nn.BatchNorm3d(
                    mid_ch
                ),

                nn.LeakyReLU(
                    0.1,
                    inplace=True,
                ),
            )
        )

        self.fc_weight1 = (
            nn.Conv3d(
                mid_ch,
                out_ch,
                kernel_size=1,
                bias=True,
            )
        )

        self.fc_weight2 = (
            nn.Conv3d(
                mid_ch,
                out_ch,
                kernel_size=1,
                bias=True,
            )
        )

        self.fc_weight3 = (
            nn.Conv3d(
                mid_ch,
                out_ch,
                kernel_size=1,
                bias=True,
            )
        )

        # =====================================================
        # Spatial competition
        # =====================================================
        self.spatial_gate = (
            nn.Sequential(

                nn.Conv3d(
                    out_ch,
                    3,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),

                nn.BatchNorm3d(
                    3
                ),
            )
        )

        self.softmax = nn.Softmax(
            dim=0
        )

        # =====================================================
        # Saliency cleaning
        # HW
        # =====================================================
        self.spatial_atten_hw = (
            nn.Sequential(

                nn.Conv2d(
                    2,
                    1,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),

                nn.Sigmoid(),
            )
        )

        # =====================================================
        # Saliency cleaning
        # DH
        # =====================================================
        self.spatial_atten_dh = (
            nn.Sequential(

                nn.Conv2d(
                    2,
                    1,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),

                nn.Sigmoid(),
            )
        )

        self.conv_y = nn.Sequential(

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_ch
            ),
        )

        self.relu = nn.LeakyReLU(
            0.1,
            inplace=True,
        )

    def forward(
        self,
        x,
    ):

        # =====================================================
        # Multi-scale extraction
        # =====================================================
        x_d1 = (
            self.branch1(
                x
            )
        )

        x_d2 = (
            self.branch2(
                x
            )
        )

        x_d3 = (
            self.branch3(
                x
            )
        )

        u = (
            x_d1
            + x_d2
            + x_d3
        )

        # =====================================================
        # 1. Channel competition
        # =====================================================
        feat_pool = (
            self.gap(
                u
            )
            +
            self.gmp(
                u
            )
        )

        feat_compact = (
            self.fc_compact(
                feat_pool
            )
        )

        w1_ch = (
            self.fc_weight1(
                feat_compact
            )
        )

        w2_ch = (
            self.fc_weight2(
                feat_compact
            )
        )

        w3_ch = (
            self.fc_weight3(
                feat_compact
            )
        )

        ch_weights = (
            self.softmax(
                torch.stack(
                    [
                        w1_ch,
                        w2_ch,
                        w3_ch,
                    ],
                    dim=0,
                )
            )
        )

        # =====================================================
        # 2. Spatial competition
        # =====================================================
        sp_weights = (
            self.spatial_gate(
                u
            )
            .unsqueeze(
                2
            )
        )

        sp_weights = (
            sp_weights.permute(
                1,
                0,
                2,
                3,
                4,
                5,
            )
        )

        sp_weights = (
            self.softmax(
                sp_weights
            )
        )

        x_prime_prime = (

            x_d1
            * ch_weights[0]
            * sp_weights[0]

            +

            x_d2
            * ch_weights[1]
            * sp_weights[1]

            +

            x_d3
            * ch_weights[2]
            * sp_weights[2]
        )

        # =====================================================
        # 3. Cross-dimensional saliency cleaning
        # =====================================================
        (
            b,
            c,
            d,
            h,
            w,
        ) = x_prime_prime.shape

        # =====================================================
        # HW plane
        # =====================================================
        x_hw = torch.cat(
            [
                x_prime_prime.mean(
                    dim=1,
                    keepdim=True,
                ),

                x_prime_prime.max(
                    dim=1,
                    keepdim=True,
                )[0],
            ],
            dim=1,
        )

        x_hw = x_hw.view(
            b * d,
            2,
            h,
            w,
        )

        M_hw = (
            self.spatial_atten_hw(
                x_hw
            )
            .view(
                b,
                1,
                d,
                h,
                w,
            )
        )

        # =====================================================
        # DH plane
        # =====================================================
        x_dh = (
            x_prime_prime
            .permute(
                0,
                4,
                1,
                2,
                3,
            )
            .contiguous()
            .view(
                b * w,
                c,
                d,
                h,
            )
        )

        dh_in = torch.cat(
            [
                x_dh.mean(
                    dim=1,
                    keepdim=True,
                ),

                x_dh.max(
                    dim=1,
                    keepdim=True,
                )[0],
            ],
            dim=1,
        )

        M_dh = (
            self.spatial_atten_dh(
                dh_in
            )
            .view(
                b,
                w,
                1,
                d,
                h,
            )
            .permute(
                0,
                2,
                3,
                4,
                1,
            )
            .contiguous()
        )

        # =====================================================
        # Saliency fusion
        # =====================================================
        M = (
            M_hw
            + M_dh
        ) / 2.0

        x_enhanced = (
            x_prime_prime
            * M
        )

        y = (
            x_prime_prime
            +
            self.conv_y(
                x_enhanced
            )
        )

        return (
            self.relu(
                y
            )
        )


# =========================================================
# 6. SpaTri-Net
# =========================================================
class SpaTri_Net(
    nn.Module
):

    def __init__(
        self,
        in_ch=1,
        num_classes=2,
    ):

        super().__init__()

        # =====================================================
        # Single-modality constraint
        # =====================================================
        if in_ch != 1:

            raise ValueError(
                "当前 SpaTri-Net 按单模态 "
                "3D sMRI 设计，"
                "in_ch 应设置为 1。"
            )

        # =====================================================
        # SPIN
        # =====================================================
        self.spi = SPIN()

        # =====================================================
        # Encoder
        #
        # MRI = 1 channel
        # XYZ = 3 channels
        # total = 4 channels
        # =====================================================
        self.enc1 = (
            ResidualBottleneck3D(
                in_ch + 3,
                16,
            )
        )
        # 4 -> 64

        self.enc2 = (
            ResidualBottleneck3D(
                64,
                32,
                stride=2,
            )
        )
        # 64 -> 128

        self.enc3 = (
            ResidualBottleneck3D(
                128,
                64,
                stride=2,
            )
        )
        # 128 -> 256

        self.enc4 = (
            ResidualBottleneck3D(
                256,
                128,
                stride=2,
            )
        )
        # 256 -> 512

        # =====================================================
        # MS-SCE
        # =====================================================
        self.bridge_aspp = (
            MS_SCE(
                in_ch=512,
                out_ch=256,
                r=4,
            )
        )

        # =====================================================
        # AC-STA
        # =====================================================
        self.triplet_att = (
            AC_STA(
                channels=256,
                kernel_size=3,
            )
        )

        # =====================================================
        # Classification
        # =====================================================
        self.gap = (
            nn.AdaptiveAvgPool3d(
                1
            )
        )

        self.classifier = (
            nn.Sequential(

                nn.Linear(
                    256,
                    128,
                ),

                nn.BatchNorm1d(
                    128
                ),

                nn.ReLU(
                    True
                ),

                nn.Dropout(
                    0.5
                ),

                nn.Linear(
                    128,
                    num_classes,
                ),
            )
        )

        # =====================================================
        # Initialization
        # =====================================================
        self._init_weights()

    # =========================================================
    # Weight initialization
    # =========================================================
    def _init_weights(
        self,
    ):

        for m in self.modules():

            if isinstance(
                m,
                (
                    nn.Conv3d,
                    nn.Conv2d,
                ),
            ):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                m,
                (
                    nn.BatchNorm3d,
                    nn.BatchNorm2d,
                ),
            ):

                nn.init.constant_(
                    m.weight,
                    1,
                )

                nn.init.constant_(
                    m.bias,
                    0,
                )

    # =========================================================
    # Forward
    # =========================================================
    def forward(
        self,
        x,
        coords,
    ):

        b = x.size(
            0
        )

        # =====================================================
        # Step 1
        # Template-space coordinate prior injection
        # =====================================================
        x = self.spi(
            x,
            coords,
        )

        # =====================================================
        # Step 2
        # Residual encoder
        # =====================================================
        x = self.enc1(
            x
        )

        x = self.enc2(
            x
        )

        x = self.enc3(
            x
        )

        x = self.enc4(
            x
        )

        # =====================================================
        # Step 3
        # MS-SCE
        # =====================================================
        x = self.bridge_aspp(
            x
        )

        # =====================================================
        # Step 4
        # AC-STA
        # =====================================================
        x = self.triplet_att(
            x
        )

        # =====================================================
        # Step 5
        # Global classification
        # =====================================================
        x = self.gap(
            x
        )

        x = x.view(
            b,
            -1,
        )

        x = self.classifier(
            x
        )

        return x


# =========================================================
# Sanity check
# =========================================================
if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SpaTri_Net(
        in_ch=1,
        num_classes=2,
    ).to(
        device
    )

    # =====================================================
    # Dummy MRI
    # =====================================================
    dummy_input = torch.randn(
        2,
        1,
        96,
        96,
        96,
        device=device,
    )

    # =====================================================
    # Dummy coordinates
    #
    # 这里只用于测试 forward。
    # 真正训练时 coords 来自 Dataset，
    # 不再在 model 中生成。
    # =====================================================
    axis_d = torch.linspace(
        -1,
        1,
        96,
        device=device,
    )

    axis_h = torch.linspace(
        -1,
        1,
        96,
        device=device,
    )

    axis_w = torch.linspace(
        -1,
        1,
        96,
        device=device,
    )

    grid_d = (
        axis_d
        .view(
            1,
            1,
            96,
            1,
            1,
        )
        .expand(
            2,
            1,
            96,
            96,
            96,
        )
    )

    grid_h = (
        axis_h
        .view(
            1,
            1,
            1,
            96,
            1,
        )
        .expand(
            2,
            1,
            96,
            96,
            96,
        )
    )

    grid_w = (
        axis_w
        .view(
            1,
            1,
            1,
            1,
            96,
        )
        .expand(
            2,
            1,
            96,
            96,
            96,
        )
    )

    dummy_coords = torch.cat(
        [
            grid_d,
            grid_h,
            grid_w,
        ],
        dim=1,
    )

    # =====================================================
    # Forward
    # =====================================================
    model.eval()

    with torch.no_grad():

        output = model(
            dummy_input,
            dummy_coords,
        )

    print(
        "=" * 70
    )

    print(
        "SpaTri-Net Template-Space "
        "SPIN Forward: SUCCESS"
    )

    print(
        "=" * 70
    )

    print(
        f"MRI input shape: "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Coordinate shape: "
        f"{tuple(dummy_coords.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M"
    )