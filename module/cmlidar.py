import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models


class ResNet34Features(nn.Module):
    def __init__(self, ):
        super(ResNet34Features, self).__init__()
        # 加载 ResNet34
        resnet = models.resnet34(weights=None)

        # 拆分各个模块
        self.conv1 = resnet.conv1  # 7x7 卷积
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # ResNet34 的 4 个大层
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        outputs = {}  # 用字典存储各层输出，方便调用

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


class GFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, mask=None):
        # x: [B, N, C]
        x = self.fc1(x)
        out = F.relu(x)
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
        self.init_pcd = nn.Linear(3, 32)
        self.pct1 = GFEBlock(32, 64)
        self.pct2 = GFEBlock(64, 128)
        self.pct3 = GFEBlock(128, 256)
        self.pct4 = GFEBlock(256, 512)

        # Cross Attention
        self.ca1 = CrossModalAttention(64, 64, 64)
        self.ca2 = CrossModalAttention(128, 128, 128)
        self.ca3 = CrossModalAttention(256, 256, 256)
        self.ca4 = CrossModalAttention(512, 512, 512)

        # 上采样解码
        self.up1 = nn.Linear(512, 256)
        self.up2 = nn.Linear(256, 128)
        self.up3 = nn.Linear(128, 64)

        self.fc_out = nn.Linear(64, 12)  # 输出 xyz

        # RGB backbone
        self.img_backbone = ResNet34Features()

    def forward(self, pc, img, mask=None):
        """
        pc: [B, N, 3]
        img: [B, 3, H, W]
        mask: [B, N]   (1=有效点, 0=padding)
        """
        B, N, _ = pc.shape

        img_feats = self.img_backbone(img)
        img_flat = [f.flatten(2).transpose(1, 2) for f in img_feats]

        # 编码
        pc = F.relu(self.init_pcd(pc))

        x1 = self.pct1(pc, mask)
        x1 = self.ca1(x1, img_flat[0], mask)

        x2 = self.pct2(x1, mask)
        x2 = self.ca2(x2, img_flat[1], mask)

        x3 = self.pct3(x2, mask)
        x3 = self.ca3(x3, img_flat[2], mask)

        x4 = self.pct4(x3, mask)
        x4 = self.ca4(x4, img_flat[3], mask)

        # 逐层上采样
        up1 = self.up1(x4) + x3
        up2 = self.up2(up1) + x2
        up3 = self.up3(up2) + x1

        # 输出点云 (4N 个点)
        out = self.fc_out(up3)  # [B, N, 3]

        # 步骤 1: 把最后的 12 reshape 成 (4, 3)
        out = out.view(out.size(0), out.size(1), 4, 3)  # (B, N, 4, 3)

        # 步骤 2: 把 N 和 4 合并
        out = out.view(out.size(0), out.size(1) * 4, 3)  # (B, 4N, 3)
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
    model = PointCloudUpsampler()
    out = model(pts, imgs, masks)
    print("输入点云:", pts.shape)
    print("输入RGB:", imgs.shape)
    print("输出点云:", out.shape)  # [B, 4*max_points, 3]
