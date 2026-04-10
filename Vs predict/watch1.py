import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import lasio
import matplotlib.pyplot as plt

# ============================
# 模型结构（完全不动）
# ============================
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
    def __init__(self, modes=12, width=32):
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

# ============================
# 🚀 预测 + 真实Vs(从SDT算) VS 预测Vs
# ============================
def predict_vs_and_plot(las_path, save_path="output.las"):
    print("🔹 开始预测...")

    device = torch.device("cpu")
    model = FNO1d().to(device)

    try:
        model.load_state_dict(torch.load("best_vs_model.pth", map_location=device))
        print("✅ 模型加载成功")
    except:
        print("❌ 模型不存在！")
        return

    model.eval()
    las = lasio.read(las_path)
    dep = las.index
    print(f"✅ 读取LAS：{len(dep)} 个深度点")

    # 安全取曲线
    def get(name):
        if name in las.curves:
            return las[name]
        else:
            return np.full_like(dep, np.nan)

    GR = get("GR")
    DT = get("DT")
    RHOB = get("RHOB")
    SDT = get("SDT")  # <-- 读入SDT

    # ======================
    # 👉 关键：从 SDT 计算 真实Vs
    # ======================
    true_Vs = np.full_like(SDT, np.nan)
    valid = ~np.isnan(SDT)
    true_Vs[valid] = 1e6 / SDT[valid]  # 真实Vs

    # 归一化
    def safe_norm(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        s = 1 if s < 1e-6 else s
        return (x - m) / s

    GR_n = safe_norm(GR)
    DT_n = safe_norm(DT)
    RHOB_n = safe_norm(RHOB)

    feat = np.stack([GR_n, DT_n, RHOB_n], axis=-1)
    feat = np.nan_to_num(feat, 0)

    # 预测 Vs
    with torch.no_grad():
        x = torch.tensor(feat[None], dtype=torch.float32).to(device)
        pred_Vs = model(x).cpu().numpy()[0]

    pred_Vs = np.clip(pred_Vs, 100, 1000)

    # 保存LAS
    las["VS_PRED"] = pred_Vs
    las["VS_TRUE"] = true_Vs
    las.write(save_path)
    print(f"✅ 预测完成！已保存到：{save_path}")

    # ============================
    # ✅ 画图：真实Vs(SDT转) VS 预测Vs
    # ============================
    plt.figure(figsize=(6, 10))

    # 只画有效点，避免NaN
    mask = ~np.isnan(true_Vs)
    plt.plot(pred_Vs[mask], dep[mask], label='Pred Vs', color='blue', linewidth=1.2)
    plt.plot(true_Vs[mask], dep[mask], label='True Vs (from SDT)', color='red', linewidth=1.2)

    plt.gca().invert_yaxis()
    plt.xlabel('Vs (m/s)')
    plt.ylabel('Depth (m)')
    plt.title('True Vs vs Predicted Vs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return dep, true_Vs, pred_Vs

# ============================
# 运行
# ============================
if __name__ == "__main__":
    predict_vs_and_plot("1/23-97.las")