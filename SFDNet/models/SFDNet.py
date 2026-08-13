import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetFeaturePropagation
)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        p1    = probs[:, 1]
        t1    = (target == 1).float()
        intersection = torch.sum(p1 * t1)
        union        = torch.sum(p1) + torch.sum(t1)
        dice_coeff   = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_coeff

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth= smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        p0, p1 = probs[:,0], probs[:,1]

        t1 = (target == 1).float()
        t0 = 1.0 - t1

        TP = torch.sum(p1 * t1)
        FP = torch.sum(p1 * t0)
        FN = torch.sum(p0 * t1)

        TI = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        loss = (1.0 - TI) ** self.gamma
        return loss

class WeightedFTDLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=1.5, smooth=1e-6, w=0.7):
        super().__init__()
        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.dice = DiceLoss(smooth=smooth)
        self.w = w

    def forward(self, logits, target):
        loss_ftl  = self.ftl(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w * loss_ftl + (1.0 - self.w) * loss_dice

class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=True, add_chn=0):
        super().__init__()
        self.normal_channel = normal_channel
        total_chn = 3 + (add_chn if normal_channel else 0)

        self.transformer1_layer = nn.TransformerEncoderLayer(
            d_model=32+64, nhead=6, dim_feedforward= (32+64)*4, dropout=0.1
        )
        self.transformer1 = nn.TransformerEncoder(
            self.transformer1_layer, num_layers=2
        )

        self.transformer2_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=256*4, dropout=0.1
        )
        self.transformer2 = nn.TransformerEncoder(
            self.transformer2_layer, num_layers=2
        )

        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.05,0.1], [16,32],
            total_chn, [[16,16,32],[32,32,64]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.1,0.2], [16,32],
            32+64, [[64,64,128],[64,96,128]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2,0.4], [16,32],
            128+128, [[128,196,256],[128,196,256]]
        )
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4,0.8], [16,32],
            256+256, [[256,256,512],[256,384,512]]
        )

        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256,256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256,256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256,128])
        self.fp1 = PointNetFeaturePropagation(128, [128,128,128])

        self.conv1 = nn.Conv1d(128,128,1)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        if self.normal_channel:
            l0_points = xyz
            l0_xyz    = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz    = xyz

        l1_xyz, l1_pts = self.sa1(l0_xyz, l0_points)
        t1 = l1_pts.permute(2,0,1)
        t1 = self.transformer1(t1)
        l1_pts = t1.permute(1,2,0)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)

        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        l4_xyz, l4_pts = self.sa4(l3_xyz, l3_pts)

        f3_in = self.fp4(l3_xyz, l4_xyz, l3_pts, l4_pts)
        t2 = f3_in.permute(2,0,1)
        t2 = self.transformer2(t2)
        f3_in = t2.permute(1,2,0)
        l2_f = self.fp3(l2_xyz, l3_xyz, l2_pts, f3_in)
        l1_f = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_f)
        l0_f = self.fp1(l0_xyz, l1_xyz, None, l1_f)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_f))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0,2,1)
        return x, l4_pts

class get_loss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=1.5, smooth=1e-6, w=0.7):
        super().__init__()
        self.loss_fn = WeightedFTDLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth, w=w)

    def forward(self, pred, target, trans_feat=None, weight=None):
        return self.loss_fn(pred, target)

if __name__ == '__main__':
    import torch
    model = get_model(2, normal_channel=True, add_chn=3)
    xyz   = torch.rand(6, 3+3, 2048)
    out,_ = model(xyz)
    print(out.shape)
