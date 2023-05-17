
import torch.nn as nn
import torch





class Resnet_block(nn.Module):
  def __init__(self,in_channels,out_channels,stride=1):
    super(Resnet_block,self).__init__()
    self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=stride,padding=1,bias=False)
    self.bn1= nn.BatchNorm2d(out_channels)
    self.relu1=nn.ReLU()
    
    self.conv2=nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
    self.bn2=nn.BatchNorm2d(out_channels)
    self.relu2=nn.ReLU()

    self.residual=nn.Sequential()
    if stride!=1 or in_channels!=out_channels:
      self.residual=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_channels)
                                  )

  def forward(self,x):
    
    out=self.conv1(x)
    out=self.bn1(out)
    out= self.relu1(out)

    out =self.conv2(out)
    out =self.bn2(out)

    out +=self.residual(x)
    out= self.relu2(out)
    return out
  
class custom_Resnet(nn.Module):
  def __init__(self,block,n_start_filters,layers):
    super(custom_Resnet,self).__init__()
    self.in_channels=n_start_filters
    self.layer1=nn.Sequential(
    nn.Conv2d(3,n_start_filters,kernel_size=3,bias=False,padding=1),
    nn.BatchNorm2d(n_start_filters),
    nn.ReLU(inplace=True),
#     nn.Dropout2d(p=0.3)
    )
    self.layer2=self.make_layer(block,n_start_filters,layers[0],stride=1)
    self.layer3=self.make_layer(block,n_start_filters*2,layers[1],stride=2)
    self.layer4=self.make_layer(block,n_start_filters*4,layers[2],stride=2)
    # self.layer4=self.make_layer(block,n_start_filters*8,layers[3],stride=2)
    # self.dropout = nn.Dropout(dropout_prob)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # self.fc = nn.Linear(self.in_channels, num_classes)
    for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    for m in self.modules():
        if isinstance(m, Resnet_block):
            nn.init.constant_(m.bn2.weight, 0)

  
  def make_layer(self,block,out_channels,n_blocks,stride):
    layers=[]
    layers.append(block(self.in_channels,out_channels,stride))
    self.in_channels=out_channels
    layers.extend([block(out_channels,out_channels) for i in range(1,n_blocks)])
    return nn.Sequential(*layers)
    # layers = [layer1, layer2, layer3]
    # model = nn.Sequential(*layers)

  def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        # out = self.fc(out)
        return out
