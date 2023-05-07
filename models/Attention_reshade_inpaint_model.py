###### Source Color Inpainting: https://github.com/vt-vl-lab/3d-photo-inpainting/blob/58bed6e2a84000902031ca108d5a9712fd72427d/networks.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, g=16, channel_att=False, spatial_att=False):
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        self.slide_winsize = in_channels * kernel_size * kernel_size

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_channels, out_channels//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_channels//g, out_channels, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0

        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = ((output - output_bias) * self.slide_winsize) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            output_pool = torch.cat([F.adaptive_avg_pool2d(output, (1, 1)), F.adaptive_max_pool2d(output, (1, 1))], dim=1)
            att = self.att_c(output_pool)
            output = output * att
        if self.spatial_att:
            output_pool = torch.cat([torch.mean(output, dim=1, keepdim=True), torch.max(output, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(output_pool)
            output = output * att

        return output, new_mask



class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False, channel_att=False, spatial_att=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias, channel_att=channel_att, spatial_att=spatial_att)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias, channel_att=channel_att, spatial_att=spatial_att)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias, channel_att=channel_att, spatial_att=spatial_att)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias, channel_att=channel_att, spatial_att=spatial_att)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class Inpaint_Color_Net(nn.Module):
    def __init__(self, layer_size=7, upsampling_mode='nearest', add_hole_mask=False, add_two_layer=False, add_border=False, channel_att=False, spatial_att=False):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        in_channels = 4
        self.enc_1 = PCBActiv(in_channels, 64, bn=False, sample='down-7', channel_att=False, spatial_att=False)
        self.enc_2 = PCBActiv(64, 128, sample='down-5', channel_att=False, spatial_att=False)
        self.enc_3 = PCBActiv(128, 256, sample='down-5', channel_att=False, spatial_att=False)
        self.enc_4 = PCBActiv(256, 512, sample='down-3', channel_att=False, spatial_att=False)
        self.enc_5 = PCBActiv(512, 512, sample='down-3', channel_att=False, spatial_att=False)
        self.enc_6 = PCBActiv(512, 512, sample='down-3', channel_att=False, spatial_att=False)
        self.enc_7 = PCBActiv(512, 512, sample='down-3', channel_att=False, spatial_att=False)

        self.dec_7 = PCBActiv(512+512, 512, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_6 = PCBActiv(512+512, 512, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)

        self.dec_5A = PCBActiv(512 + 512, 512, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_4A = PCBActiv(512 + 256, 256, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_3A = PCBActiv(256 + 128, 128, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_2A = PCBActiv(128 + 64, 64, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_1A = PCBActiv(64 + in_channels, 3, bn=False, activ=None, conv_bias=True)
        '''
        self.dec_5B = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4B = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3B = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2B = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1B = PCBActiv(64 + 4, 1, bn=False, activ=None, conv_bias=True)
        '''
    def cat(self, A, B):
        return torch.cat((A, B), dim=1)

    def upsample(self, feat, mask):
        feat = F.interpolate(feat, scale_factor=2, mode=self.upsampling_mode)
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        return feat, mask

    def forward_3P(self, mask, context, rgb, edge, unit_length=128, cuda=None):
        with torch.no_grad():
            input = torch.cat((rgb, edge, context, mask), dim=1)
            n, c, h, w = input.shape
            residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h) # + 128
            residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w) # + 256
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input = torch.zeros((n, c, h + residual_h, w + residual_w)).to(cuda)
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input
            # enlarge_input[:, 3] = 1. - enlarge_input[:, 3]
            enlarge_input = enlarge_input.to(cuda)
            rgb_output = self.forward(enlarge_input)
            rgb_output = rgb_output[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]

        return rgb_output

    def forward(self, input, add_border=False):
        input_mask = (input[:, -1:]).clamp(0, 1)
        H, W = input.shape[-2:]
        f_0, h_0 = input, input_mask.repeat((1,input.shape[1],1,1))
        f_1, h_1 = self.enc_1(f_0, h_0)
        f_2, h_2 = self.enc_2(f_1, h_1)
        f_3, h_3 = self.enc_3(f_2, h_2)
        f_4, h_4 = self.enc_4(f_3, h_3)
        f_5, h_5 = self.enc_5(f_4, h_4)
        f_6, h_6 = self.enc_6(f_5, h_5)
        f_7, h_7 = self.enc_7(f_6, h_6)

        o_7, k_7 = self.upsample(f_7, h_7)
        o_6, k_6 = self.dec_7(self.cat(o_7, f_6), self.cat(k_7, h_6))
        o_6, k_6 = self.upsample(o_6, k_6)
        o_5, k_5 = self.dec_6(self.cat(o_6, f_5), self.cat(k_6, h_5))
        o_5, k_5 = self.upsample(o_5, k_5)
        o_5A, k_5A = o_5, k_5
        o_5B, k_5B = o_5, k_5
        ###############
        o_4A, k_4A = self.dec_5A(self.cat(o_5A, f_4), self.cat(k_5A, h_4))
        o_4A, k_4A = self.upsample(o_4A, k_4A)
        o_3A, k_3A = self.dec_4A(self.cat(o_4A, f_3), self.cat(k_4A, h_3))
        o_3A, k_3A = self.upsample(o_3A, k_3A)
        o_2A, k_2A = self.dec_3A(self.cat(o_3A, f_2), self.cat(k_3A, h_2))
        o_2A, k_2A = self.upsample(o_2A, k_2A)
        o_1A, k_1A = self.dec_2A(self.cat(o_2A, f_1), self.cat(k_2A, h_1))
        o_1A, k_1A = self.upsample(o_1A, k_1A)
        o_0A, k_0A = self.dec_1A(self.cat(o_1A, f_0), self.cat(k_1A, h_0))

        return torch.sigmoid(o_0A)


class Inpaint_Color_Net_lite(nn.Module):
    def __init__(self, layer_size=7, upsampling_mode='nearest', add_hole_mask=False, add_two_layer=False, add_border=False, channel_att=False, spatial_att=False, depth_aware=0):
        super(Inpaint_Color_Net_lite, self).__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size       

        in_channels = 4
        if depth_aware:
            in_channels+=1
        self.enc_1 = PCBActiv(in_channels, 32, bn=False, sample='down-7', channel_att=False, spatial_att=False)
        self.enc_2 = PCBActiv(32, 64, sample='down-5', channel_att=False, spatial_att=False)
        self.enc_3 = PCBActiv(64, 128, sample='down-5', channel_att=False, spatial_att=False)
        self.enc_4 = PCBActiv(128, 256, sample='down-3', channel_att=False, spatial_att=False)
        self.enc_5 = PCBActiv(256, 256, sample='down-3', channel_att=False, spatial_att=False)
        self.enc_6 = PCBActiv(256, 256, sample='down-3', channel_att=False, spatial_att=False)
        self.enc_7 = PCBActiv(256, 256, sample='down-3', channel_att=False, spatial_att=False)

        self.dec_7 = PCBActiv(256+256, 256, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_6 = PCBActiv(256+256, 256, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)

        self.dec_5A = PCBActiv(256 + 256, 256, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_4A = PCBActiv(256 + 128, 128, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_3A = PCBActiv(128 + 64, 64, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_2A = PCBActiv(64 + 32, 32, activ='leaky', channel_att=channel_att, spatial_att=spatial_att)
        self.dec_1A = PCBActiv(32 + in_channels, 3, bn=False, activ=None, conv_bias=True)
        '''
        self.dec_5B = PCBActiv(512 + 512, 512, activ='leaky')
        self.dec_4B = PCBActiv(512 + 256, 256, activ='leaky')
        self.dec_3B = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_2B = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_1B = PCBActiv(64 + 4, 1, bn=False, activ=None, conv_bias=True)
        '''
    def cat(self, A, B):
        return torch.cat((A, B), dim=1)

    def upsample(self, feat, mask):
        feat = F.interpolate(feat, scale_factor=2, mode=self.upsampling_mode)
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        return feat, mask

    def forward_3P(self, mask, context, rgb, edge, unit_length=128, cuda=None):
        with torch.no_grad():
            input = torch.cat((rgb, edge, context, mask), dim=1)
            n, c, h, w = input.shape
            residual_h = int(np.ceil(h / float(unit_length)) * unit_length - h) # + 128
            residual_w = int(np.ceil(w / float(unit_length)) * unit_length - w) # + 256
            anchor_h = residual_h//2
            anchor_w = residual_w//2
            enlarge_input = torch.zeros((n, c, h + residual_h, w + residual_w)).to(cuda)
            enlarge_input[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w] = input
            # enlarge_input[:, 3] = 1. - enlarge_input[:, 3]
            enlarge_input = enlarge_input.to(cuda)
            rgb_output = self.forward(enlarge_input)
            rgb_output = rgb_output[..., anchor_h:anchor_h+h, anchor_w:anchor_w+w]

        return rgb_output

    def forward(self, input, add_border=False):
        input_mask = (input[:, -1:]).clamp(0, 1)
        H, W = input.shape[-2:]
        f_0, h_0 = input, input_mask.repeat((1,input.shape[1],1,1))
        f_1, h_1 = self.enc_1(f_0, h_0)
        f_2, h_2 = self.enc_2(f_1, h_1)
        f_3, h_3 = self.enc_3(f_2, h_2)
        f_4, h_4 = self.enc_4(f_3, h_3)
        f_5, h_5 = self.enc_5(f_4, h_4)
        f_6, h_6 = self.enc_6(f_5, h_5)
        f_7, h_7 = self.enc_7(f_6, h_6)

        o_7, k_7 = self.upsample(f_7, h_7)
        o_6, k_6 = self.dec_7(self.cat(o_7, f_6), self.cat(k_7, h_6))
        o_6, k_6 = self.upsample(o_6, k_6)
        o_5, k_5 = self.dec_6(self.cat(o_6, f_5), self.cat(k_6, h_5))
        o_5, k_5 = self.upsample(o_5, k_5)
        o_5A, k_5A = o_5, k_5
        o_5B, k_5B = o_5, k_5
        ###############
        o_4A, k_4A = self.dec_5A(self.cat(o_5A, f_4), self.cat(k_5A, h_4))
        o_4A, k_4A = self.upsample(o_4A, k_4A)
        o_3A, k_3A = self.dec_4A(self.cat(o_4A, f_3), self.cat(k_4A, h_3))
        o_3A, k_3A = self.upsample(o_3A, k_3A)
        o_2A, k_2A = self.dec_3A(self.cat(o_3A, f_2), self.cat(k_3A, h_2))
        o_2A, k_2A = self.upsample(o_2A, k_2A)
        o_1A, k_1A = self.dec_2A(self.cat(o_2A, f_1), self.cat(k_2A, h_1))
        o_1A, k_1A = self.upsample(o_1A, k_1A)
        o_0A, k_0A = self.dec_1A(self.cat(o_1A, f_0), self.cat(k_1A, h_0))

        return torch.sigmoid(o_0A)