# DenseNet-BC (3 DENSE BLOCKS)
class DenseUnit(torch.nn.Module):               # BN-ReLU-Conv(1×1) > 4k > -BN-ReLU-Conv(3×3)
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
    self.relu = torch.nn.ReLU()
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
    self.relu = torch.nn.ReLU()
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

class DenseNet_Pro(torch.nn.Module):              # conv_in > MaxPool > BLK1 > TS1 > BLK2 > TS2 > BLK3 > AvgPool > FC
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
    ts1_in = 2*growth_rate + growth_rate*(blk_num_list[0])                                           
    self.ts1 = DenseTs(ts1_in, theta, drop_rate)

    blk2_in = int(ts1_in*theta)                       
    self.denseblk2 = DenseBLK(blk2_in, growth_rate, blk_num_list[1], drop_rate)
    ts2_in = blk2_in + growth_rate*(blk_num_list[1])
    self.ts2 = DenseTs(ts2_in, theta, drop_rate)

    blk3_in = int(ts2_in*theta)
    self.denseblk3 = DenseBLK(blk3_in, growth_rate, blk_num_list[2], drop_rate) #if ts3 > after H*W = 2*2 

    self.bn_out = torch.nn.BatchNorm2d(blk3_in + growth_rate*blk_num_list[2])
    self.relu = torch.nn.ReLU()
    self.pool_out = torch.nn.AdaptiveAvgPool2d((1, 1))
    self.linear = torch.nn.Linear(blk3_in + growth_rate*blk_num_list[2],10)

  def forward(self,x):
    out = self.conv_in(x)

    out = self.denseblk1(out)
    out = self.ts1(out)

    out = self.denseblk2(out)
    out = self.ts2(out)

    out = self.denseblk3(out)
    
    out = self.relu(self.bn_out(out))
    out = self.pool_out(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out
 ### mod = DenseNet_Pro(12,[16,16,16],0.5,0.2) # DenseNet-BC 100
