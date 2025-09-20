import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_load import LiDARnpzDataset
from module.cmlidar import PointCloudUpsampler

from torchvision import transforms


# =====================
# 3. 训练函数
# =====================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        x = batch["x"].to(device)  # [B, N, 3]
        y = batch["y"].to(device)  # [B, N, 3]
        x_mask = batch["x_mask"].to(device)  # [B, N]
        y_mask = batch["y_mask"].to(device)  # [B, N]
        rgb = batch["rgb"].to(device)  # [B, N]

        optimizer.zero_grad()
        pred = model(x, rgb, x_mask)  # [B, N, 3]

        # 只计算 mask=1 的点的损失
        loss = chamfer_distance(pred * y_mask.unsqueeze(-1), y * y_mask.unsqueeze(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------------------- Loss Function ----------------------------
def chamfer_distance(pc1, pc2):
    # pc1, pc2: (B, N, 3)
    diff1 = torch.cdist(pc1, pc2)  # (B, N, N)
    diff2 = diff1.transpose(1, 2)
    min_dist1 = torch.min(diff1, dim=2)[0]
    min_dist2 = torch.min(diff2, dim=2)[0]
    return (torch.mean(min_dist1) + torch.mean(min_dist2))


# =====================
# 4. 主程序
# =====================
def main():
    dataset_root = "D:\WorkSpace\PycharmProjects\CM-LiDAR\datasets"  # 修改为你的 npz 数据集目录
    log_file = 'training_log.csv'
    vit_input_sizes = {
        'vit_small': (224, 224),
        'vit_base': (384, 384),
        'vit_large': (512, 512)
    }
    selected_size = vit_input_sizes['vit_large']  # 可选择 'vit_small', 'vit_base', 'vit_large'

    transform = transforms.Compose([
        transforms.Resize(selected_size,antialias=True),
    ])

    dataset = LiDARnpzDataset(dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # 模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointCloudUpsampler().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    with open(log_file, 'a', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Epoch', 'Time(s)', 'Loss'])

        # 训练循环
        num_epochs = 200
        best_loss = float('inf')  # 初始化最优loss

        for epoch in range(num_epochs):
            start_time = time.time()
            loss = train_one_epoch(model, dataloader, optimizer, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.6f}")

            epoch_time = time.time() - start_time
            avg_loss = loss / len(dataloader)
            log_writer.writerow([epoch, epoch_time, avg_loss])
            # 保存最优模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved at epoch {epoch} with loss {best_loss:.6f}")


if __name__ == "__main__":
    main()
