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


def evaluate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch["x"].to(device)  # [B, N, 3]
            y = batch["y"].to(device)  # [B, N, 3]
            x_mask = batch["x_mask"].to(device)  # [B, N]
            y_mask = batch["y_mask"].to(device)  # [B, N]
            rgb = batch["rgb"].to(device)  # [B, 3, H, W]

            pred = model(x, rgb, x_mask)  # [B, N, 3]

            # 损失只计算有效点
            loss = chamfer_distance(pred * y_mask.unsqueeze(-1), y * y_mask.unsqueeze(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# =====================
# 4. 主程序
# =====================
def main():
    dataset_root = "D:\WorkSpace\PycharmProjects\CM-LiDAR\d"  # 修改为你的 npz 数据集目录
    log_file = 'training_log.csv'
    train_ratio = 0.6
    vit_input_sizes = {
        'vit_small': (224, 224),
        'vit_base': (384, 384),
        'vit_large': (512, 512)
    }
    selected_size = vit_input_sizes['vit_base']  # 可选择 'vit_small', 'vit_base', 'vit_large'

    transform = transforms.Compose([
        transforms.Resize(selected_size, antialias=True),
    ])

    train_dataset = LiDARnpzDataset(dataset_root, train=True, train_ratio=train_ratio,
                                    transform=transform)
    test_dataset = LiDARnpzDataset(dataset_root, train=False, train_ratio=train_ratio,
                                   transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 模型、损失函数、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointCloudUpsampler().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    with open(log_file, 'a', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Epoch', 'Time(s)', 'Train Loss', 'Test Loss'])

        # 训练循环
        num_epochs = 200
        best_loss = float('inf')  # 初始化最优loss

        for epoch in range(num_epochs):
            start_time = time.time()

            # 训练
            train_loss = train_one_epoch(model, train_dataloader, optimizer, device)
            epoch_time = time.time() - start_time
            avg_train_loss = train_loss / len(train_dataloader)

            # 测试
            test_loss = evaluate_one_epoch(model, test_dataloader, device)
            avg_test_loss = test_loss / len(test_dataloader)

            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {avg_test_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s")

            # 记录日志
            log_writer.writerow([epoch, epoch_time, avg_train_loss, avg_test_loss])

            # 保存最优模型（按测试集损失）
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
    main()
