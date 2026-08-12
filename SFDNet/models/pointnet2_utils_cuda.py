import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils as _p2


def fps_sample(xyz_bn3, npoint):
    return _p2.furthest_point_sample(xyz_bn3.contiguous(), npoint)


def gather_bcn(features_bcn, idx_bs):
    return _p2.gather_operation(features_bcn.contiguous(), idx_bs.int())


def group_bcn(features_bcn, idx_bsk):
    return _p2.grouping_operation(features_bcn.contiguous(), idx_bsk.int())


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz_coord = xyz[:, :3, :].contiguous()
        B, _, N = xyz_coord.shape
        S = self.npoint

        xyz_bn3 = xyz_coord.permute(0, 2, 1).contiguous()

        fps_idx = fps_sample(xyz_bn3, S)
        new_xyz = gather_bcn(xyz_coord, fps_idx)
        new_xyz_bn3 = new_xyz.permute(0, 2, 1).contiguous()

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = _p2.ball_query(radius, K, xyz_bn3, new_xyz_bn3)
            grouped_xyz = group_bcn(xyz_coord, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)

            if points is not None:
                grouped_points = group_bcn(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=1)
            else:
                grouped_points = grouped_xyz

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 3)[0]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1_bn3 = xyz1[:, :3, :].permute(0, 2, 1).contiguous()
        xyz2_bn3 = xyz2[:, :3, :].permute(0, 2, 1).contiguous()
        B, N, _ = xyz1_bn3.shape
        S = xyz2_bn3.shape[1]

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dist, idx = _p2.three_nn(xyz1_bn3, xyz2_bn3)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = _p2.three_interpolate(
                points2.contiguous(), idx.int(), weight)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz_coord = xyz[:, :3, :].contiguous()
        B, _, N = xyz_coord.shape

        if self.group_all:
            new_xyz = torch.zeros(B, 3, 1, device=xyz.device, dtype=xyz.dtype)
            grouped_xyz = xyz_coord.unsqueeze(2)
            if points is not None:
                grouped_points = torch.cat(
                    [points.unsqueeze(2), grouped_xyz], dim=1)
            else:
                grouped_points = grouped_xyz
        else:
            S = self.npoint
            K = self.nsample
            xyz_bn3 = xyz_coord.permute(0, 2, 1).contiguous()
            fps_idx = fps_sample(xyz_bn3, S)
            new_xyz = gather_bcn(xyz_coord, fps_idx)
            new_xyz_bn3 = new_xyz.permute(0, 2, 1).contiguous()
            group_idx = _p2.ball_query(self.radius, K, xyz_bn3, new_xyz_bn3)
            grouped_xyz = group_bcn(xyz_coord, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            if points is not None:
                grouped_points = group_bcn(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=1)
            else:
                grouped_points = grouped_xyz

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))
        new_points = torch.max(grouped_points, 3)[0]
        return new_xyz, new_points
