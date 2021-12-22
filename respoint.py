import torch.nn as nn
import torch.nn.functional as F
import torch
from models.PointNetSetAbstractionMsg import PointNetSetAbstractionMsg, PointNetSetAbstraction


class ResPoint(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(ResPoint,self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64,64], [64, 64, 128,128], [96, 96, 128,128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128,128], [128, 128, 256,256], [128, 128,256, 256]])
        self.sa3 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 640,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        # self.sa4 = PointNetSetAbstraction(None, None, None, 320 + 3, [256, 512, 1024], True)
        self.sc4 = PointNetFeatureConcat()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm) #[8,3,512] [8,320,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #[8,3 128] [8,640,512]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[8,3,1] [8,1024,1]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # l4_xyz, l4_points = self.sc4(l3_xyz,l4_xyz, l3_points, l4_points)
        x = l4_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l4_points

class PointNetFeatureConcat(nn.Module):
    def __init__(self):
        super(PointNetFeatureConcat, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

    def forward(self, xyz1, xyz2, points1, points2):#[8,3 128]  [8,3,1] [8,640,512] [8,1024,1]
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        new_points = []
        return new_points
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


