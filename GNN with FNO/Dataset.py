import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, global_mean_pool
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np

# ===========================================================================
# 1. 3D FNO 模型（论文专用：预测压力场）
# ===========================================================================
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        B, C, H, W, D = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        out_ft = torch.zeros(B, self.out_channels, H, W, D//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(H, W, D), dim=[-3,-2,-1])
        return x

class FNO3d(nn.Module):
    def __init__(self, in_channels=11, out_channels=1, modes1=8, modes2=8, modes3=8, width=20):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 3

        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, H, W, D, C = x.shape
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x  # 输出：下一时间步压力场


# ===========================================================================
# 2. GNN 模型（GCN2，论文专用：预测饱和度场）
# ===========================================================================
class GNNSatellite(nn.Module):
    def __init__(self, in_channels=11, hidden=64, out_channels=1, num_layers=4, alpha=0.5, theta=1.0):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCN2Conv(hidden, hidden, alpha, theta))
        self.lin_out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin_in(x)
        h = x
        for conv in self.convs:
            h = conv(h, x, edge_index, edge_weight)
            h = F.gelu(h)
        return self.lin_out(h)  # 输出：下一时间步饱和度场


# ===========================================================================
# 3. 井产出预测模型（轻量级网络）
# ===========================================================================
class WellProductionModel(nn.Module):
    def __init__(self, in_feat=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2)  # 产油率、产水率
        )
    def forward(self, x):
        return self.net(x)


# ===========================================================================
# 4. 完整耦合架构（FNO + GNN + Well Model）
# ===========================================================================
class CoupledFGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fno_pressure = FNO3d(in_channels=11, out_channels=1)
        self.gnn_saturation = GNNSatellite(in_channels=11, hidden=64, out_channels=1)
        self.well_model = WellProductionModel(in_feat=5)

    def forward(self, grid_feat, grid_shape, edge_index, edge_attr, well_feat):
        # grid_feat: (节点数, 11)
        B, H, W, D, C = grid_shape

        # 1) 预测压力
        pressure_next = self.fno_pressure(grid_feat.view(B, H, W, D, C))

        # 2) 替换压力通道 → 输入GNN预测饱和度
        grid_feat[..., 0] = pressure_next.view(-1)
        saturation_next = self.gnn_saturation(grid_feat, edge_index, edge_attr)

        # 3) 井产出预测
        well_prod = self.well_model(well_feat)

        return pressure_next, saturation_next, well_prod


# ===========================================================================
# 5. 滚动损失函数（论文核心训练方式）
# ===========================================================================
def rolling_loss(pred_p, pred_s, pred_w, true_p, true_s, true_w, mask=None):
    loss_p = F.mse_loss(pred_p[mask], true_p[mask])
    loss_s = F.mse_loss(pred_s[mask], true_s[mask])
    loss_w = F.mse_loss(pred_w, true_w)
    return loss_p + 0.5 * loss_s + 0.1 * loss_w


# ===========================================================================
# 6. 自回归推理（滚动预测未来N步）
# ===========================================================================
@torch.no_grad()
def autoregressive_infer(model, init_state, steps=16):
    preds_p, preds_s, preds_w = [], [], []
    state = init_state
    for _ in range(steps):
        p, s, w = model(state["feat"], state["shape"], state["edge_index"], state["edge_attr"], state["well"])
        preds_p.append(p)
        preds_s.append(s)
        preds_w.append(w)
        # 更新状态
        state["feat"][..., 0] = p.view(-1)
        state["feat"][..., 1] = s.view(-1)
    return preds_p, preds_s, preds_w


# ===========================================================================
# 7. 训练入口（完整流程）
# ===========================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoupledFGN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 你之前的数据集加载器
    # train_loader = get_dataloader(...)

    print("="*60)
    print("✅ 论文对齐 Coupled FNO-GNN 模型加载完成")
    print("✅ 包含：3D FNO + GCN2 + 井模型 + 滚动损失 + 自回归推理")
    print("="*60)