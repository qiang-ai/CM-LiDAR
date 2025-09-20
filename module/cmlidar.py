import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# -------------------------
# 简化版 ResNet backbone
# -------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = [base_channels, base_channels * 2, base_channels * 4]

    def forward(self, x):
        f1 = self.layer1(x)  # [B, C, H/2, W/2]
        f2 = self.layer2(f1)  # [B, 2C, H/4, W/4]
        f3 = self.layer3(f2)  # [B, 4C, H/8, W/8]
        return [f1, f2, f3]


# -------------------------
# PCT Block (Point Transformer)
# -------------------------
class PCTBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, mask=None):
        # x: [B, N, C]
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.norm(out + x)  # 残差
        if mask is not None:
            out = out * mask.unsqueeze(-1)  # 掩码
        return out


# -------------------------
# Cross-Attention 融合 RGB特征
# -------------------------
class CrossModalAttention(nn.Module):
    def __init__(self, pc_dim, img_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(pc_dim, hidden_dim)
        self.key = nn.Linear(img_dim, hidden_dim)
        self.value = nn.Linear(img_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, pc_dim)

    def forward(self, pc_feat, img_feat, mask=None):
        # pc_feat: [B, N, C1]
        # img_feat: [B, HW, C2]
        Q = self.query(pc_feat)  # [B, N, H]
        K = self.key(img_feat)  # [B, HW, H]
        V = self.value(img_feat)  # [B, HW, H]
        attn = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)  # [B, N, HW]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # [B, N, H]
        out = self.out(out) + pc_feat
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


# -------------------------
# U-Net 式 PCT+ResNet 融合网络
# -------------------------
class PointCloudUpsampler(nn.Module):
    def __init__(self, img_dim=64, hidden_dim=128):
        super().__init__()
        # PCT编码
        self.init_pcd = nn.Linear(3, hidden_dim)
        self.pct1 = PCTBlock(hidden_dim, hidden_dim)
        self.pct2 = PCTBlock(hidden_dim, hidden_dim)
        self.pct3 = PCTBlock(hidden_dim, hidden_dim)

        # Cross Attention
        self.ca1 = CrossModalAttention(hidden_dim, img_dim, hidden_dim)
        self.ca2 = CrossModalAttention(hidden_dim, img_dim * 2, hidden_dim)
        self.ca3 = CrossModalAttention(hidden_dim, img_dim * 4, hidden_dim)

        # 上采样解码
        self.up1 = nn.Linear(hidden_dim, hidden_dim)
        self.up2 = nn.Linear(hidden_dim, hidden_dim)
        self.up3 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 3)  # 输出 xyz

        # RGB backbone
        self.img_backbone = ResNetBackbone()

    def forward(self, pc, img, mask=None):
        """
        pc: [B, N, 3]
        img: [B, 3, H, W]
        mask: [B, N]   (1=有效点, 0=padding)
        """
        B, N, _ = pc.shape

        img_feats = self.img_backbone(img)  # [f1, f2, f3]
        img_flat = [f.flatten(2).transpose(1, 2) for f in img_feats]

        # 编码
        pc = F.relu(self.init_pcd(pc))

        x1 = self.pct1(pc, mask)
        x1 = self.ca1(x1, img_flat[0], mask)

        x2 = self.pct2(x1, mask)
        x2 = self.ca2(x2, img_flat[1], mask)

        x3 = self.pct3(x2, mask)
        x3 = self.ca3(x3, img_flat[2], mask)

        # 逐层上采样
        up1 = self.up1(x3) + x2
        up2 = self.up2(up1) + x1
        up3 = self.up3(up2)

        # 输出点云 (4N 个点)
        out = self.fc_out(up3)  # [B, N, 3]
        out = out.repeat_interleave(4, dim=1)  # 扩展为 4N
        if mask is not None:
            mask = mask.repeat_interleave(4, dim=1)
            out = out * mask.unsqueeze(-1)
        return out


# -------------------------
# 测试代码 (动态点数 + mask)
# -------------------------
if __name__ == "__main__":
    B = 8
    max_points = 1500
    pts = []
    masks = []
    for _ in range(B):
        n = torch.randint(1000, 1500, (1,)).item()  # 动态点数
        pc = torch.randn(n, 3)
        mask = torch.ones(n)
        # padding 到 max_points
        pad_len = max_points - n
        if pad_len > 0:
            pc = torch.cat([pc, torch.zeros(pad_len, 3)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len)], dim=0)
        pts.append(pc)
        masks.append(mask)

    pts = torch.stack(pts).float()  # [B, max_points, 3]
    masks = torch.stack(masks).float()  # [B, max_points]

    imgs = torch.randn(B, 3, 384, 384)
    vit_input_sizes = {
        'vit_small': (224, 224),
        'vit_base': (384, 384),
        'vit_large': (512, 512)
    }
    selected_size = vit_input_sizes['vit_large']  # 可选择 'vit_small', 'vit_base', 'vit_large'

    transform = transforms.Compose([
        transforms.Resize(selected_size),
        transforms.ToTensor()
    ])
    model = PointCloudUpsampler(transform=transform)
    out = model(pts, imgs, masks)
    print("输入点云:", pts.shape)
    print("输入RGB:", imgs.shape)
    print("输出点云:", out.shape)  # [B, 4*max_points, 3]
