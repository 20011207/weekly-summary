import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
import matplotlib.pyplot as plt
import lasio
import random
from torch.utils.data import Dataset, DataLoader

# ========================
# 固定随机种子（保证可复现）
# ========================
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ========================
# 配置
# ========================
folder_path = r'1'
TARGET_STEP = 0.125
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1      # 井长不一样，必须用 batch=1
EPOCHS = 500
LR = 5e-4

# ========================
# 读取 LAS
# ========================
las_file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".las")]
all_las = []
for f in las_file_list:
    try:
        las = lasio.read(os.path.join(folder_path, f))
        all_las.append(las)
        print(f"✅ {f}")
    except:
        print(f"❌ {f}")

# ========================
# 全局归一化（非常关键！）
# ========================
all_GR, all_DT, all_RHOB, all_Vs = [], [], [], []  # 新增 all_Vs 存储原始Vs
for las in all_las:
    try:
        all_GR.extend(las['GR'])
        all_DT.extend(las['DT'])
        all_RHOB.extend(las['RHOB'])
        all_Vs.extend(las['Vs'])  # 直接读取Vs列
    except:
        pass

mean_gr, std_gr = np.nanmean(all_GR), np.nanstd(all_GR)
mean_dt, std_dt = np.nanmean(all_DT), np.nanstd(all_DT)
mean_rhob, std_rhob = np.nanmean(all_RHOB), np.nanstd(all_RHOB)
# 新增Vs的归一化参数（可选，根据需求决定是否归一化Vs）
mean_vs, std_vs = np.nanmean(all_Vs), np.nanstd(all_Vs)

# ========================
# 数据集（不统一长度！FNO 原生支持）
# ========================
class WellDataset(Dataset):
    def __init__(self, las_list):
        self.list = las_list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        las = self.list[idx]
        dep = las.index

        def get(kk):
            return las[kk] if kk in las.keys() else np.full_like(dep, np.nan)

        GR   = (get("GR") - mean_gr) / std_gr
        DT   = (get("DT") - mean_dt) / std_dt
        RHOB = (get("RHOB") - mean_rhob) / std_rhob
        Vs   = get("Vs")  # 直接读取Vs，替代原来的SDT

        feat = np.stack([GR, DT, RHOB], axis=-1)
        feat = np.nan_to_num(feat, 0)

        # 处理Vs的掩码（过滤NaN值）
        mask = ~np.isnan(Vs)
        Vs = np.nan_to_num(Vs, 0)  # NaN值填充为0（训练时通过mask屏蔽）
        mask = mask.astype(np.float32)

        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(Vs, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )

# ========================
# 模型（小模型！防止过拟合）
# ========================
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(out_channels, in_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.weights.shape[0], L//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=L)

class FNO1d(nn.Module):
    def __init__(self, modes=12, width=32):  # 变小！防过拟合
        super().__init__()
        self.fc0 = nn.Linear(3, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 1)

    def forward(self, x):
        x = self.fc0(x).permute(0,2,1)
        x = torch.relu(self.conv0(x) + self.w0(x))
        x = torch.relu(self.conv1(x) + self.w1(x))
        return self.fc1(x.permute(0,2,1)).squeeze(-1)

# ========================
# 训练
# ========================
model = FNO1d().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

ds = WellDataset(all_las)
dl = DataLoader(ds, batch_size=1, shuffle=True)

print("\n开始训练...")
best_loss = 1e9

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for feat, vs, mask in dl:
        feat, vs, mask = feat.to(DEVICE), vs.to(DEVICE), mask.to(DEVICE)
        pred = model(feat)
        loss = criterion(pred*mask, vs*mask)  # 只计算有效Vs的损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dl)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_vs_model.pth")

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

print("\n✅ 训练完成！best model saved")

# ========================
# 画图（直接比对真实Vs和预测Vs）
# ========================
model.load_state_dict(torch.load("best_vs_model.pth"))
model.eval()

with torch.no_grad():
    # 取第一个井的数据进行可视化
    feat, true_vs, mask = ds[0]
    pred_vs = model(feat.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

    # 只展示有效数据（非NaN部分）
    mask_np = mask.numpy().astype(bool)
    true_vs_valid = true_vs.numpy()[mask_np]
    pred_vs_valid = pred_vs[mask_np]
    depth_idx = np.arange(len(true_vs))[mask_np]  # 深度索引（仅有效部分）

plt.figure(figsize=(10, 6))
plt.plot(depth_idx, true_vs_valid, label='True Vs', color='blue', linewidth=1.5)
plt.plot(depth_idx, pred_vs_valid, label='Pred Vs', color='red', alpha=0.7, linewidth=1.5)
plt.xlabel('Depth Index')
plt.ylabel('Vs Value')
plt.title('True Vs vs Predicted Vs Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()