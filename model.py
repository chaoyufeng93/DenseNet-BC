import torch
import os
import math
import numpy as np
import torch.nn.functional as F

# DenseNet-BC (4 DENSE BLOCKS)
class DenseUnit(torch.nn.Module):               # BN-ReLU-Conv(1×1) > 4*growth_rate > -BN-ReLU-Conv(3×3)
  def __init__(self,c_in,growth_rate,idx,drop_rate):
    super(DenseUnit,self).__init__()
    self.bn1 = torch.nn.BatchNorm2d(c_in)
    self.conv1 = torch.nn.Conv2d(in_channels = c_in, out_channels = 4*growth_rate, kernel_size = 1,
                                 stride = 1, padding = 0)
    self.bn3 = torch.nn.BatchNorm2d(4*growth_rate)
    self.relu = torch.nn.ReLU(inplace = True)
    self.conv3 = torch.nn.Conv2d(in_channels = 4*growth_rate, out_channels = growth_rate, kernel_size = 3,
                                 stride = 1, padding = 1)
    self.dropout = torch.nn.Dropout(p = drop_rate)
  def forward(self,x):
    DU_out = self.relu(self.bn1(x))
    DU_out = self.conv1(DU_out)
    DU_out = self.dropout(DU_out)
    DU_out = self.relu(self.bn3(DU_out))
    DU_out = self.conv3(DU_out)
    DU_out = self.dropout(DU_out)
    DU_out = torch.cat([x,DU_out],dim=1)
    return DU_out
    

class DenseBLK(torch.nn.Module):
  def __init__(self,c_in,growth_rate,unit_num, drop_rate):
    super(DenseBLK,self).__init__()
    self.dsbn1 = torch.nn.BatchNorm2d(c_in)
    self.relu = torch.nn.ReLU(inplace=True)
    self.dense_in_conv1 = torch.nn.Conv2d(in_channels = c_in, out_channels = 4*growth_rate, kernel_size = 1,
                                 stride = 1, padding = 0)
    self.dropout = torch.nn.Dropout(p = drop_rate)
    self.dsbn3 = torch.nn.BatchNorm2d(4*growth_rate)
    self.dense_in_conv3 = torch.nn.Conv2d(in_channels = 4*growth_rate, out_channels = growth_rate, kernel_size = 3,
                                 stride = 1, padding = 1)
    self.connect = torch.nn.ModuleList([DenseUnit(c_in + i*growth_rate, growth_rate, i, drop_rate = drop_rate) for i in range(1,unit_num)])
    
  def forward(self,x):
    BLK_out = self.relu(self.dsbn1(x))
    BLK_out = self.dense_in_conv1(BLK_out)
    BLK_out = self.dropout(BLK_out)
    BLK_out = self.relu(self.dsbn3(BLK_out))
    BLK_out = self.dense_in_conv3(BLK_out)
    BLK_out = self.dropout(BLK_out)
    BLK_out = torch.cat([x,BLK_out],dim=1)
    for i, fuc in enumerate(self.connect):
      BLK_out = self.connect[i](BLK_out)
    return BLK_out
    

class DenseTs(torch.nn.Module):
  def __init__(self,c_in,theta,drop_rate):
    super(DenseTs,self).__init__()
    self.bn1 = torch.nn.BatchNorm2d(c_in)
    self.relu = torch.nn.ReLU(inplace=True)
    self.conv1 = torch.nn.Conv2d(in_channels = c_in, out_channels = int(c_in*theta), kernel_size = 1,
                                 stride = 1, padding = 0)
    self.dropout = torch.nn.Dropout(p = drop_rate)
    
    self.pool = torch.nn.AvgPool2d(kernel_size=2)

  def forward(self,x):               
    out = self.relu(self.bn1(x))
    out = self.conv1(out)
    out = self.dropout(out)
    out = self.pool(out)
    return out
    

class DenseNet_Pro(torch.nn.Module):              # conv_in > MaxPool > BLK1 > TS1 > BLK2 > TS2 > BLK3 > TS3 > BLK4 > AvgPool > FC
  def __init__(self, growth_rate,blk_num_list,theta,drop_rate):
    super(DenseNet_Pro,self).__init__()

    self.grl = growth_rate
    self.bnl = blk_num_list
    self.theta = theta
    self.drop_rate = drop_rate

    self.conv_in = torch.nn.Conv2d(in_channels = 3, out_channels = 2*self.grl, kernel_size = 3,
                                 stride = 1, padding = 1)
    #self.bn_in = torch.nn.BatchNorm2d(2*self.grl)
    #self.pool_in = torch.nn.MaxPool2d(kernel_size = 2)
    self.relu = torch.nn.ReLU(inplace=True)

    self.denseblk1 = DenseBLK(2*self.grl, self.grl, self.bnl[0], self.drop_rate)
    self.ts1_in = 2*self.grl + self.grl*(self.bnl[0])                                           
    self.ts1 = DenseTs(self.ts1_in, self.theta,self.drop_rate)

    self.blk2_in = int(self.theta*self.ts1_in)                       
    self.denseblk2 = DenseBLK(self.blk2_in, self.grl, self.bnl[1],self.drop_rate)
    self.ts2_in = self.blk2_in + self.grl*(self.bnl[1])
    self.ts2 = DenseTs(self.ts2_in, self.theta,self.drop_rate)

    self.blk3_in = int(self.ts2_in*self.theta)
    self.denseblk3 = DenseBLK(self.blk3_in, self.grl, self.bnl[2],self.drop_rate) #if ts3 > after H*W = 2*2 
    self.ts3_in = self.blk3_in + self.grl*self.bnl[2]
    self.ts3 = DenseTs(self.ts3_in, self.theta,self.drop_rate)

    self.blk4_in = int(self.ts3_in*self.theta)
    self.denseblk4 = DenseBLK(self.blk4_in, self.grl, self.bnl[3],self.drop_rate)

    self.bn_out = torch.nn.BatchNorm2d(self.blk4_in + self.grl*self.bnl[3])
    self.pool_out = torch.nn.AdaptiveAvgPool2d((1, 1))
    self.linear = torch.nn.Linear(self.blk4_in + self.grl*self.bnl[3],10)  #denseblk = 3 : if conv_in padding = 1 then 2*2*(int(self.ts2_in*self.theta)+ self.grl[2]*(self.bnl[2]))
                                                                              #denseblk = 4 : self.blk4_in + self.grl[3]*self.bnl[3]
  def forward(self,x):
   # out = self.pool_in(self.relu(self.bn_in(self.conv_in(x)))) #### this for ImageNet
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
