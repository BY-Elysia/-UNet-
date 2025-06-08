import torch.nn as nn
import torch
import torch.nn.functional as F


from nets.deform_conv_v2 import DeformConv2d
from nets.tasks import TransformerDecoder
from torchsummary import summary

# 返回相应的激活函数，否则为relu()
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

# 创建一个由多个卷积层组成的序列
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

# 一个二维卷积层，一个批量归一化层，一个可激活函数 *********
class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        # if(out.shape[0] == 1):
        #     print(1)
        out = self.norm(out)
        return self.activation(out)

# 通过最大池化和一系列卷积操作来降低输入数据的维度，同时对其进行特征提取 ******
class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

# 构建一个具有上采样和多个卷积层的神经网络块
class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)  # 简单地通过插值进行上采样
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)  # 反卷积进行上采样
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)

        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

# 通道维度上的注意力机制，通过全局平均池化和最大池化提取特征图的空间信息
class CoordAtt(nn.Module):
    def __init__(self, inp1,inp2, oup, reduction=4):
        super(CoordAtt, self).__init__()
        mip = inp1 // reduction
        self.conv1 = nn.Conv2d(inp1, inp1 // reduction, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inp1 // reduction)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inp2, inp1 // reduction, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(inp1 // reduction)
        self.relu2 = nn.ReLU()
        self.conv_d = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(inp2, inp1, kernel_size=1, stride=1, padding=0)
    def forward(self, g, x):
        b, c, h, w = x.size()

        g_h = F.adaptive_avg_pool2d(g, (h, 1))
        g_w = F.adaptive_avg_pool2d(g, (1, w)).permute(0, 1,3,2)

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1,3,2)

        g_y = torch.cat([g_h, g_w], dim=2)
        g_y = self.conv1(g_y)
        g_y = self.bn1(g_y)
        g_y = self.relu1(g_y)

        x_y = torch.cat([x_h, x_w], dim=2)
        x_y = self.conv2(x_y)
        x_y = self.bn2(x_y)
        x_y = self.relu2(x_y)

        g_h, g_w = torch.split(g_y, [h, w], dim=2)
        g_w = g_w.permute(0, 1, 3,2)
        x_h, x_w = torch.split(x_y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = (x_h + g_h) / 2
        a_w = (x_w + g_w) / 2

        a_h, a_w =  torch.sigmoid(self.conv_h(a_h)), torch.sigmoid(self.conv_w(a_w))
        # x= self.conv_x(x)
        x = x * a_h * a_w
        return x

# 转置卷积上采样，并将特征图与下采样路径中的特征图拼接*****
class UpBlockAlig(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlockAlig, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.cca = CoordAtt3(in_channels//2)
    def forward(self, x, skip_x):
        out = self.up(x)
        skip_x = self.cca(skip_x, out)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
        # return self.nConvs(skip_x)


# # 不同变体的 U-Net 模型
# class UNetBasic(nn.Module):
#     def __init__(self, n_channels=3, n_classes=9):
#         '''
#         n_channels : number of channels of the input.
#                         By default 3, because we have RGB images
#         n_labels : number of channels of the ouput.
#                       By default 3 (2 labels + 1 for the background)
#         '''
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#
#         in_channels = 64
#         self.inc = ConvBatchNorm(n_channels, in_channels)
#         self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
#         self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
#         self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
#         self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
#         self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
#         self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
#         self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
#         self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
#         self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))
#
#         self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
#         self.maxpoll1 = nn.AdaptiveMaxPool2d((1, 1))
#         self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
#         self.maxpoll2 = nn.AdaptiveMaxPool2d((1, 1))
#
#         self.fc1 = nn.Linear(in_channels*8,in_channels*4)
#         self.fc2 = nn.Linear(in_channels*4,1)
#
#         if n_classes == 1:
#             self.last_activation = nn.Sigmoid()
#         else:
#             self.last_activation = None
#
#     def forward(self, x):
#         # Question here
#         x = x.float()
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#
#         x4 = self.down3(x3)
#
#         # cl_out1_a = self.avgpool1(x4)
#         # cl_out1_m = self.maxpoll1(x4)
#         x5 = self.down4(x4)
#
#         # cl_out_2_cl, x5 = self.task2(x5, x5)
#         cl_out2_a = self.avgpool2(x5)
#         # cl_out2_m = self.maxpoll2(x5)
#
#
#         x = self.up4(x5, x4)
#         x = self.up3(x, x3)
#         x = self.up2(x, x2)
#         x = self.up1(x, x1)
#         # cl_out = torch.flatten(torch.cat([cl_out1_a, cl_out1_m, cl_out2_a, cl_out2_m], dim=1), 1)
#         cl_out = torch.flatten( cl_out2_a,1)
#
#         cl_out = self.fc1(cl_out)
#         cl_out = self.fc2(cl_out)
#
#
#         logits = self.outc(x)
#         return logits, cl_out

# 坐标注意力机制，增强特征表示能力
class CoordAtt3(nn.Module):
    def __init__(self, inp):
        super(CoordAtt3, self).__init__()
        self.conv1_e = _make_nConv(inp, inp, 1, 'ReLU')
        self.conv2_e = _make_nConv(inp, inp, 1, 'ReLU')
        self.avgpool_e = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool_e = nn.AdaptiveMaxPool2d((1,1))
        # self.soft_pool_e = AdaptiveSoftPool2d(output_size=(1, 1))
        self.fc_avg = nn.Conv2d(inp,inp//2,kernel_size=1, stride=1, padding=0)
        self.fc_max = nn.Conv2d(inp,inp//2,kernel_size=1, stride=1, padding=0)
        self.fc_soft = nn.Conv2d(inp,inp//2,kernel_size=1, stride=1, padding=0)
        self.fc_avg_max_sfot = nn.Conv2d(inp//2,inp,kernel_size=1, stride=1, padding=0)
        self.deformabel = DeformConv2d(in_channels=inp, out_channels=inp,kernel_size=3)

    def forward(self, e, d):
        e_1 = self.conv1_e(e)
        avg = self.avgpool_e(e_1)
        max = self.maxpool_e(e_1)

        fc_avg = self.fc_avg(avg)
        fc_max = self.fc_max(max)

        avg_max = F.relu(fc_avg)+F.relu(fc_max)

        avg_max_soft = F.sigmoid(self.fc_avg_max_sfot(avg_max))

        def_d = self.conv2_e(d)
        avg_max_soft_def = avg_max_soft*def_d
        out = e_1+ avg_max_soft_def+def_d

        return out

# 不同变体的 U-Net 模型  *********
class UNetTaskAligWeight1(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_classes : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.down5_= DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)  ### 自定义
        self.up4 = UpBlockAlig(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlockAlig(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlockAlig(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlockAlig(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))

        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.task2 = TransformerDecoder(dim=in_channels * 8, depth=1, heads=8, dim_head=64,  #####
                                        mlp_dim=2048, dropout=0,decoder_pos_size=14,
                                         softmax=True)
        # self.task2 = TransformerDecoder(dim=1472, depth=1, heads=8, dim_head=64,
        #                                 mlp_dim=2048, dropout=0,
        #                                 decoder_pos_size=224, softmax=True)
        self.fc1 = nn.Linear(in_channels * 8, in_channels * 4)
        self.fc2 = nn.Linear(in_channels * 4, 3)  # 三个数值输出的任务#####
        # 使用 Adaptive Pooling 将不同尺寸的特征图都调整为 [4, C, 14, 14]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.conv = nn.Conv2d(in_channels*(1+2+4+8+8), 512, kernel_size=1)
    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        out0 = self.down4(x4)
        # x1, x2, x3, x4, out0
        # torch.Size([4, 64, 224, 224])
        # torch.Size([4, 128, 112, 112])
        # torch.Size([4, 256, 56, 56])
        # torch.Size([4, 512, 28, 28])
        # torch.Size([4, 512, 14, 14])

        out1 = self.up4(out0, x4)
        out2 = self.up3(out1, x3)
        out3 = self.up2(out2, x2)
        out4 = self.up1(out3, x1)
        # out1, out2, out3, out4
        # torch.Size([4, 256, 28, 28])
        # torch.Size([4, 128, 56, 56])
        # torch.Size([4, 64, 112, 112])
        # torch.Size([4, 64, 224, 224])

        # 自定义
        x1 = self.adaptive_pool(x1)  # [4, 64, 14, 14]
        x2 = self.adaptive_pool(x2)  # [4, 128, 14, 14]
        x3 = self.adaptive_pool(x3)  # [4, 256, 14, 14]
        x4 = self.adaptive_pool(x4)  # [4, 512, 14, 14]
        out0_ = self.adaptive_pool(out0)  # [4, 512, 14, 14]

        # 拼接后尺寸：[4, 64+128+256+512+512, 224, 224] = [4, 1472, 14, 14]
        dense_out = torch.cat([x1, x2, x3, x4, out0_], dim=1)  # dim=1 表示在通道维度上进行拼接
        # 通过卷积调整最终的通道数
        out0 = self.conv(dense_out)  # [4, 512, 14, 14]
        # out0   :torch.Size([4, 512, 14, 14])
        cl_out_2_cl, out0 = self.task2(out0, out0)
        # cl_out2_a = self.avgpool2(cl_out_2_cl)

        # cl_out = torch.flatten(cl_out2_a,1)
        # cl_out = self.fc1(cl_out)
        # cl_out = self.fc2(cl_out)   ##########
        # cl_out = torch.softmax(self.fc2(cl_out), dim=1)  # ########
        logits = self.outc(out4)
        #      se_out, cl_out
        return logits


if __name__ == '__main__':
    # 创建一个示例模型
    model = UNetTaskAligWeight()
    print(model)
    # 如果有 GPU 可用，则将模型移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # # 打印模型的详细信息
    # # 输入形状 (3, 224, 224) 对应于彩色图像 (C, H, W)
    # summary(model, (3, 224, 224))
