import torch 
import numpy as np 
import math 

def get_SH_basis(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    
    numElem = normal.size()[-2]* normal.size()[-1]

    norm_X = normal[:, 0].view(normal.shape[0], -1)
    norm_Y = normal[:, 1].view(normal.shape[0], -1)
    norm_Z = normal[:,2].view(normal.shape[0], -1)
    
    pi = math.pi
    #att= pi*np.array([1., 2.0/3.0, 1/4.0])
    sh_basis = torch.zeros((normal.shape[0], numElem, 9)).cuda()

    
    sh_basis[:,0] = -0.2820948*1#0.5/np.sqrt(pi)
    sh_basis[:,1] = -0.3257350*norm_Y
    sh_basis[:,2] = -0.3257350*norm_Z
    sh_basis[:,3] = -0.3257350*norm_X

    sh_basis[:,4] = 0.2731371*norm_Y*norm_X
    sh_basis[:,5] = -0.2731371*norm_Y*norm_Z
    sh_basis[:,6] = 0.136586*norm_Z**2-0.0788479
    sh_basis[:,7] = -0.1931371*norm_X*norm_Z
    sh_basis[:,8] = 0.136586*(norm_X**2-norm_Y**2)
    return sh_basis.detach()

def get_SH_basis_no_coeff(normal):
    '''
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    
    numElem = normal.size()[-2]* normal.size()[-1]

    norm_X = normal[:, 0].view(-1)
    norm_Y = normal[:, 1].view(-1)
    norm_Z = normal[:,2].view(-1)
    
    pi = math.pi
    #att= pi*np.array([1., 2.0/3.0, 1/4.0])
    sh_basis = torch.zeros((numElem, 9)).cuda()

    
    sh_basis[:,0] = 1#0.5/np.sqrt(pi)
    sh_basis[:,1] = norm_Y
    sh_basis[:,2] = norm_Z
    sh_basis[:,3] = norm_X

    sh_basis[:,4] = norm_Y*norm_X
    sh_basis[:,5] = norm_Y*norm_Z
    sh_basis[:,6] = 3*norm_Z**2-1
    sh_basis[:,7] = norm_X*norm_Z
    sh_basis[:,8] = (norm_X**2-norm_Y**2)
    return sh_basis.detach()

def least_square_solver_SHCoeff(SH_basis_src, spred):
    X = SH_basis_src
    Y = spred.mean(1).view(-1).unsqueeze(-1) #+ gpred.mean(1).view(-1).unsqueeze(-1)
    XtX, XtY = X.permute(1,0).mm(X), X.permute(1,0).mm(Y)
    XtXinv = torch.inverse(XtX + 1e-6*torch.eye(9).cuda())
    SH_coeff = torch.mm(XtXinv, XtY)
    return SH_coeff