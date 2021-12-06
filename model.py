import torch
import os
import math
import numpy as np
import torch.nn.functional as F

# DenseNet-BC (4 DENSE BLOCKS)
class DenseUnit(torch.nn.Module):               # BN-ReLU-Conv(1×1) > (4*growth_rate) > -BN-ReLU-Conv(3×3)
  def __init__(self,c_in,growth_rate,drop_rate):
    super(DenseUnit,self).__init__()
    self.bn1 = torch.nn.BatchNorm2d(c_in)
    self.conv1 = torch.nn.Conv2d(
        in_channels = c_in,
        out_channels = 4*growth_rate,
        kernel_size = 1,
        stride = 1,
        padding = 0
    )
    self.bn3 = torch.nn.BatchNorm2d(4*growth_rate)
    self.relu = torch.nn.ReLU(inplace = True)
    self.conv3 = torch.nn.Conv2d(
        in_channels = 4*growth_rate,
        out_channels = growth_rate,
        kernel_size = 3,
        stride = 1,
        padding = 1
    )
    self.dropout = torch.nn.Dropout(p = drop_rate)

  def forward(self,x):
    DU_out = self.relu(self.bn1(x))
    DU_out = self.dropout(self.conv1(DU_out))
    DU_out = self.relu(self.bn3(DU_out))
    DU_out = self.dropout(self.conv3(DU_out))
    DU_out = torch.cat([x,DU_out],dim=1)
    return DU_out

class DenseBLK(torch.nn.Module):
  def __init__(self,c_in,growth_rate,unit_num, drop_rate):
    super(DenseBLK,self).__init__()
    self.unit_num = unit_num
    self.connect = torch.nn.ModuleList([
                                        DenseUnit(
                                            c_in + i*growth_rate,
                                            growth_rate,
                                            drop_rate = drop_rate
                                        ) for i in range(unit_num)
    ])

  def forward(self,x):
    BLK_out = self.connect[0](x)
    for i in range(1, self.unit_num):
      BLK_out = self.connect[i](BLK_out)
    return BLK_out

class DenseTs(torch.nn.Module):
  def __init__(self,c_in,theta,drop_rate):
    super(DenseTs,self).__init__()
    self.bn1 = torch.nn.BatchNorm2d(c_in)
    self.relu = torch.nn.ReLU(inplace=True)
    self.conv1 = torch.nn.Conv2d(
        in_channels = c_in,
        out_channels = int(c_in*theta),
        kernel_size = 1,
        stride = 1,
        padding = 0
    )
    self.dropout = torch.nn.Dropout(p = drop_rate)
    self.pool = torch.nn.AvgPool2d(kernel_size=2)

  def forward(self,x):               
    out = self.relu(self.bn1(x))
    out = self.dropout(self.conv1(out))
    out = self.pool(out)
    return out

class DenseNet_Pro(torch.nn.Module):              # conv_in > MaxPool > BLK1 > TS1 > BLK2 > TS2 > BLK3 > TS3 > BLK4 > AvgPool > FC
  def __init__(self, growth_rate, blk_num_list,theta, drop_rate):
    super(DenseNet_Pro,self).__init__()
    self.conv_in = torch.nn.Conv2d(
        in_channels = 3,
        out_channels = 2*growth_rate,
        kernel_size = 3,
        stride = 1,
        padding = 1
    )
    self.denseblk1 = DenseBLK(2*growth_rate, growth_rate, blk_num_list[0], drop_rate)
    self.ts1_in = 2*growth_rate + growth_rate*(blk_num_list[0])                                           
    self.ts1 = DenseTs(self.ts1_in, theta, drop_rate)

    self.blk2_in = int(self.ts1_in*theta)                       
    self.denseblk2 = DenseBLK(self.blk2_in, growth_rate, blk_num_list[1], drop_rate)
    self.ts2_in = self.blk2_in + growth_rate*(blk_num_list[1])
    self.ts2 = DenseTs(self.ts2_in, theta, drop_rate)

    self.blk3_in = int(self.ts2_in*theta)
    self.denseblk3 = DenseBLK(self.blk3_in, growth_rate, blk_num_list[2], drop_rate) #if ts3 > after H*W = 2*2 
    self.ts3_in = self.blk3_in + growth_rate*blk_num_list[2]
    self.ts3 = DenseTs(self.ts3_in, theta, drop_rate)

    self.blk4_in = int(self.ts3_in*theta)
    self.denseblk4 = DenseBLK(self.blk4_in, growth_rate, blk_num_list[3], drop_rate)

    self.bn_out = torch.nn.BatchNorm2d(self.blk4_in + growth_rate*blk_num_list[3])
    self.relu = torch.nn.ReLU()
    self.pool_out = torch.nn.AdaptiveAvgPool2d((1, 1))
    self.linear = torch.nn.Linear(self.blk4_in + growth_rate*blk_num_list[3],10) 
    
  def forward(self,x):
    # if use for ImageNet, then the first in should be: self.max_pooling(self.conv_in(x)) (224*224 > 112*112(conv) > 56*56(pooling))
    out = self.conv_in(x)

    out = self.denseblk1(out)
    out = self.ts1(out)

    out = self.denseblk2(out)
    out = self.ts2(out)

    out = self.denseblk3(out)
    out = self.ts3(out)

    out = self.denseblk4(out)
    
    out = self.relu(self.bn_out(out))
    out = self.pool_out(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out
    
