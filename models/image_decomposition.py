from __future__ import print_function

import os
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch
import torch.optim
from torchvision import transforms, datasets, models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.FloatTensor


clampmax=0.95
def color(batch):
    return (torch.cat([batch, batch, batch], dim=1))

def intensity(batch):
    if len(batch.size())==4:
        return (1/3)*torch.sum(batch, dim=1).view(batch.size()[0], 1, batch.size()[2], batch.size()[3])
    else:
        return (1/3)*torch.sum(batch, dim=1).view(batch.size()[0], 1)
    
def decompose(batch, allmodel):
    apred, spred, mplist, gpred, mslist, gscores, residual=allmodel(batch)
    dcl=list([apred])+list([spred])+list([mplist])+list([gpred])
    return dcl, mslist, gscores, residual

def compose(dclt):

    mlist=dclt[2]
    if len(mlist)>0:
        mtot=mlist[0]
        for i in range(1, len(mlist)):
            mtot=mtot*mlist[i]
        wim=dclt[0]*dclt[1]*mtot+dclt[3]
    else:
        wim=dclt[0]*dclt[1]+dclt[3]
    return wim.clamp(max=clampmax)

class smoother(nn.Module):
    def __init__(self):
        super(smoother, self).__init__()
        # trained on 3x64x64
        self.idim1=64
        self.idim2=32
        self.idim3=32
        self.idim4=32
        self.conv1 = nn.Conv2d(3, self.idim1, kernel_size=5, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.idim1)
        self.conv2 = nn.Conv2d(self.idim1, self.idim2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(self.idim2)
        self.conv3 = nn.Conv2d(self.idim2, self.idim3, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.idim3)
        self.conv4 = nn.Conv2d(self.idim3, self.idim4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(self.idim4)
        # so what I have is 8x8
        self.linear=nn.Linear(self.idim4*8*8, 5) #asmg  
#        self.bn5=nn.BatchNorm1d(5)
        self.nonlin=nn.LeakyReLU(0.2, inplace=True)
        self.lsm=nn.LogSoftmax(dim=1)

    def forward(self, x):
        conv1=self.bn1(self.nonlin(self.conv1(x)))
        conv2=self.bn2(self.nonlin(self.conv2(conv1)))
        conv3=self.bn3(self.nonlin(self.conv3(conv2)))
        conv4=self.bn4(self.nonlin(self.conv4(conv3)))
        #.view(-1, self.idim4*8*8)
#        print(conv4.size())
#        preds=self.linear(conv4)
        return conv1, conv2, conv3, conv4
    #preds  #  


### When Using Color and Big Res model
class uplayer(nn.Module):
    def __init__(self, idim, odim, subsample):
        super(uplayer, self).__init__()
        self.leaky=nn.LeakyReLU()        
        if idim>odim:
            self.conv = nn.Conv2d(idim, idim, kernel_size=5, stride=1, padding=2, bias=True)
            self.bn = nn.BatchNorm2d(idim)
            self.proj=nn.Conv2d(idim, odim, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conv = nn.Conv2d(idim, odim, kernel_size=5, stride=1, padding=2, bias=True)
            self.bn = nn.BatchNorm2d(odim)
        if subsample:
            self.ssl=nn.Conv2d(odim, odim, kernel_size=5, stride=2, padding=1, bias=False)
        self.idim=idim
        self.odim=odim
        self.subsample=subsample

    def forward(self, x):
        if self.idim>self.odim:
            conv = self.leaky(self.bn(self.conv(x)))
            conv+=x #
            conv=self.proj(conv)
        else:
            conv = self.leaky(self.bn(self.conv(x)))
            conv[:, 0:self.idim, :, :]+=x
        if self.subsample:
            conv=self.ssl(conv)
        return conv
            
class downlayer(nn.Module):
    def __init__(self, idim, odim, cdim):
        super(downlayer, self).__init__()
        self.leaky=nn.LeakyReLU()
        cidim=idim+cdim
        if cidim>odim:
            self.conv = nn.Conv2d(cidim, cidim, kernel_size=5, stride=1, padding=2, bias=True)
            self.bn = nn.BatchNorm2d(cidim)
            self.proj=nn.Conv2d(cidim, odim, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conv = nn.Conv2d(cidim, odim, kernel_size=5, stride=1, padding=2, bias=True)
            self.bn = nn.BatchNorm2d(odim)
        self.idim=idim
        self.odim=odim
        self.cdim=cdim
        self.cidim=cidim

    def forward(self, x, cross, outsize1, outsize2):
        checkdim=cross.size()[1]
        if checkdim!=self.cdim:
            print(checkdim, self.cdim)
            barf()
#        print(x.size(), cross.size())    

        wt=torch.cat([x, cross], dim=1)
        if self.cidim>self.odim:
            conv = self.leaky(self.bn(self.conv(wt)))
            conv+=wt #
            conv=self.proj(conv)
        else:
            conv = self.leaky(self.bn(self.conv(wt)))
            conv[:, 0:self.cidim, :, :]+=wt
        # we always upsample
        conv=F.interpolate(conv, size=[outsize1, outsize2], mode='bilinear')
        return conv
    
class genhead(nn.Module):
    def __init__(self, monoflag, idim6, idim5, idim4, idim3, idim2, idim1):
        super(genhead, self).__init__()        
        self.idim6=idim6
        self.idim5=idim5
        self.idim4=idim4
        self.idim3=idim3
        self.idim2=idim2
        self.idim1=idim1        
                # true if monochrome
        self.dl5=downlayer(self.idim6, self.idim5, self.idim6)
        self.dl4=downlayer(self.idim5, self.idim4, self.idim5)
        self.dl3=downlayer(self.idim4, self.idim3, self.idim4)
        self.dl2=downlayer(self.idim3, self.idim2, self.idim3)
        self.dl1=downlayer(self.idim2, self.idim1, self.idim2)
        if monoflag:
            self.decode = nn.ConvTranspose2d(2*self.idim1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.decode = nn.ConvTranspose2d(2*self.idim1, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.monoflag=monoflag

    def forward(self, c6, c5, c4, c3, c2, c1):
        dc5=self.dl5(c6, c6, c5.size()[2], c5.size()[3])
        dc4=self.dl4(dc5, c5, c4.size()[2], c4.size()[3])
        dc3=self.dl3(dc4, c4, c3.size()[2], c3.size()[3])
        dc2=self.dl2(dc3, c3, c2.size()[2], c2.size()[3])
        dc1=self.dl1(dc2, c2, c1.size()[2], c1.size()[3])
        wfeat=torch.cat([c1, dc1], dim=1)
        decode=self.decode(wfeat)
        # if it's monochrome, you have to deal with that later
        return decode

class t1head(nn.Module):
    def __init__(self, idim6, idim5, idim4, idim3, idim2, idim1):
        super(t1head, self).__init__()                
        self.idim6=idim6
        self.idim5=idim5
        self.idim4=idim4
        self.idim3=idim3
        self.idim2=idim2
        self.idim1=idim1        
        self.dl5=downlayer(self.idim6, self.idim5, self.idim6)
        self.colorest=nn.Conv2d(self.idim6, 3, kernel_size=2, stride=1, padding=1, bias=True)
        self.dl4=downlayer(self.idim5, self.idim4, self.idim5)
        self.dl3=downlayer(self.idim4, self.idim3, self.idim4)
        self.dl2=downlayer(self.idim3, self.idim2, self.idim3)
        self.dl1=downlayer(self.idim2, self.idim1, self.idim2)
        self.decode = nn.ConvTranspose2d(2*self.idim1, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, c6, c5, c4, c3, c2, c1):
        dc5=self.dl5(c6, c6, c5.size()[2], c5.size()[3])
        spc=torch.mean(self.colorest(c6), dim=[2, 3])
        dc4=self.dl4(dc5, c5, c4.size()[2], c4.size()[3])
        dc3=self.dl3(dc4, c4, c3.size()[2], c3.size()[3])
        dc2=self.dl2(dc3, c3, c2.size()[2], c2.size()[3])
        dc1=self.dl1(dc2, c2, c1.size()[2], c1.size()[3])
        wfeat=torch.cat([c1, dc1], dim=1)
        decode=color(self.decode(wfeat))
        spc=spc.view([spc.size()[0], spc.size()[1], 1, 1])
#        print(decode.size(), spc.size())
        decode=torch.mul(spc, decode) # broadcasting should allow this to work?
        return decode


class mlcrossenc(nn.Module):
    def __init__(self, nmat, device):
        super(mlcrossenc, self).__init__()
        self.idim1=32
        self.idim2=16
        self.idim3=16
        self.idim4=32
        self.idim5=64
        self.idim6=256
#        self.idim1=16
#        self.idim2=32
#        self.idim3=64
#        self.idim4=64
#        self.idim5=128
#        self.idim6=128
        # Encoder
        # input is 128
        self.ul1=uplayer(3, self.idim1, False)
        self.ul2=uplayer(self.idim1, self.idim2, True)
        self.ul3=uplayer(self.idim2, self.idim3, True)
        self.ul4=uplayer(self.idim3, self.idim4, True)
        self.ul5=uplayer(self.idim4, self.idim5, True)
        self.ul6=uplayer(self.idim5, self.idim6, True)
        self.ahead=genhead(False, self.idim6, self.idim5, self.idim4, self.idim3, self.idim2, self.idim1)
        self.shead=t1head(self.idim6, self.idim5, self.idim4, self.idim3, self.idim2, self.idim1)
        self.mlist=list()
        for i in range(0, nmat):
            mhead=genhead(True, self.idim6, self.idim5, self.idim4, self.idim3, self.idim2, self.idim1).to(device)
            self.mlist=self.mlist+list([mhead])
        self.ghead=genhead(True, self.idim6, self.idim5, self.idim4, self.idim3, self.idim2, self.idim1)        
        self.sigmoid=nn.Sigmoid()
        self.nmat=nmat
        self.nonlin=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        conv1=self.ul1(x)
        conv2=self.ul2(conv1)
        conv3=self.ul3(conv2)
        conv4=self.ul4(conv3)
        conv5=self.ul5(conv4)
        conv6=self.ul6(conv5)
        apred=self.ahead(conv6, conv5, conv4, conv3, conv2, conv1)
        spred=self.shead(conv6, conv5, conv4, conv3, conv2, conv1)
        mslist=list()
        mplist=list()        
        for i in range(0, self.nmat):
            scorer=self.mlist[i]
            mscores=scorer(conv6, conv5, conv4, conv3, conv2, conv1)
            #mpred=color(torch.sigmoid(mscores))
            mpred=color(torch.ones_like(mscores)-self.nonlin(mscores))
            mslist=mslist+list([mscores])
            mplist=mplist+list([mpred])
        gscores=self.ghead(conv6, conv5, conv4, conv3, conv2, conv1)
        gpred=color(torch.sigmoid(gscores))
        if self.nmat>0:
            mtot=mplist[0]
            for i in range(1, self.nmat):
                mtot=mtot*mplist[i]
            residual=x-(apred*spred*mtot+gpred).clamp(max=clampmax)
        else:
            residual=x-(apred*spred+gpred).clamp(max=clampmax)
        return apred, spred, mplist, gpred, mslist, gscores, residual