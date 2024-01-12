
import numpy as np
import torch
from torch import nn
F = torch.nn.functional


GPU = True


def argmax2d(arr):
    n, m = arr.shape
    ij = arr.argmax()
    i, j = ij//m, ij%m
    return i, j


def argmaxminrnd(v, maximize, rnd=True, thr=0.):
    '''Returns an index k for which v[k] max/min. If more than one index is
    max/minimizing, a random index of these is chosen.
    maximize: If True, finds the maximizing index, otherwise the minimizing index
    rnd: If False, this function will work like np.argmax/min
    thr: Small value that lowers the bound for a value to be considered as optimal.'''
    s = +1 if maximize else -1
    sv = s * np.array(v)
    if not rnd:
        return np.argmax(sv)
    else:
        available = (sv >= np.max(sv)-thr)
        inds = np.arange(len(sv))[available]
        return np.random.choice(inds)
    



def np2t(*args):
    'Converts a numpy array to a torch array'
    res = [torch.from_numpy(np.array(x, dtype='float32')) for x in args]
    if GPU:
        res = [x.cuda() for x in res]

    if len(res)==1:
        return res[0]
    else:
        return res


def t2np(*args):
    'Converts a torch array to a numpy array'
    res = [x.detach().cpu().numpy() for x in args]

    if len(res)==1:
        return res[0]
    else:
        return res


def setLearningRate(optim, lr):
    for grp in optim.param_groups:
        grp['lr'] = lr


def load_checkpoint(model, path):
    'Load with mismatched layer sizes'
    load_dict = torch.load(path)
    model_dict = model.state_dict()
    for k in model_dict.keys():
        if k in load_dict:
            if load_dict[k].shape == model_dict[k].shape:
                model_dict[k] = load_dict[k]
            else:
                print('Ignoring (since shape mismatch)', k)
        else:
            print('Ignoring (since not in data)', k)
    model.load_state_dict(model_dict)



#### Batchgenerator ####

def batchgenerator_v2(indarray, readout_xy, batchsize=32, permute=True):
    indarray = np.asarray(indarray)

    numind = len(indarray)
    while True:
        if permute:
            perm = np.random.permutation(numind)
        else:
            perm = np.arange(numind)
        for i in range((numind)//batchsize):
            inds = indarray[perm[i*batchsize:(i+1)*batchsize]]
            xs, ys = readout_xy(inds)
            yield xs, ys


def batchgenerator_v2_single(indarray, readout_xy, batchsize=32, permute=True):
    # for only xs output (instead of xs and ys)
    indarray = np.asarray(indarray)

    numind = len(indarray)
    while True:
        if permute:
            perm = np.random.permutation(numind)
        else:
            perm = np.arange(numind)
        for i in range((numind)//batchsize):
            inds = indarray[perm[i*batchsize:(i+1)*batchsize]]
            #xs, ys = readout_xy(inds)
            xs = readout_xy(inds)
            yield xs


def getbatchgen(indarray, readout_xy, batchsize=32, permute=True, single=False):
    '''Wrapper of the batchgenerator, which additionally gives the number of steps per epoche.
    This value is needed by the model.fit_generator function.'''
    perepoch = len(indarray) // batchsize
    if single:
        return batchgenerator_v2_single(indarray, readout_xy, batchsize, permute), perepoch
    else:
        return batchgenerator_v2(indarray, readout_xy, batchsize, permute), perepoch


def augment8(x):
    r = np.random.rand
    if r()<.5:
        x = x[:,::-1]
    if r()<.5:
        x = x[:,:,::-1]
    if r()<.5:
        x = x.swapaxes(2,1)
    return x


def augment8_dual(x, y):
    r = np.random.rand
    if r()<.5:
        x, y = x[:,::-1], y[:,::-1]
    if r()<.5:
        x, y = x[:,:,::-1], y[:,:,::-1]
    if r()<.5:
        x, y = x.swapaxes(2,1), y.swapaxes(2,1)
    return x, y



#### HSV and RGB ####

def rgb2hsv(img):
    from colorsys import rgb_to_hsv
    shape = img.shape
    img = img.reshape(-1, 3)
    for k in range(img.shape[0]):
        img[k] = rgb_to_hsv(*img[k])
    return img.reshape(shape)
def hsv2rgb(img):
    from colorsys import hsv_to_rgb
    shape = img.shape
    img = img.reshape(-1, 3)
    for k in range(img.shape[0]):
        img[k] = hsv_to_rgb(*img[k])
    return img.reshape(shape)


#### Blurring ####


def blurr_imgs(imgs, radius):
    from scipy import signal
    # bhwc
    if radius == 0:
        return imgs
    n = int(2.5*radius)*2+1 # odd number important!
    kernel1D = signal.gaussian(n, std=radius)
    kernel = np.outer(kernel1D, kernel1D)
    kernel = kernel[None, :, :, None] # add dummy channel
    kernel = kernel / kernel.sum()
    imgs = signal.convolve(imgs, kernel, mode='full')
    imgs = imgs[:, n//2:-n//2+1, n//2:-n//2+1]
    return imgs


_tkernels = {}
def tkernel(radius, c):
    global _tkernels
    if (radius,c) not in _tkernels:
        n = int(np.ceil(2.5*radius)*2+1) # odd number important!
        kernel1D = signal.gaussian(n, std=radius)
        kernel2D = np.outer(kernel1D, kernel1D)
        kernel2D = kernel2D / kernel2D.sum()
        tkernel = torch.zeros(c, 1, n, n)
        for i in range(c):
            tkernel[i,0] = torch.from_numpy(kernel2D)
        tkernel = tkernel.cuda()
        _tkernels[radius,c] = tkernel
    return _tkernels[radius,c]

def tblurr(y, r=2):
    y = y.permute([0, 3, 1, 2]) # nhwc to nchw
    b, c, h, w = y.shape
    ##n = tkernel(r).shape[-1] # kernel size
    ##y=torch.nn.functional.pad(y, (0, 0, n//2, n//2), mode='reflect')  ## padding
    return torch.nn.functional.conv2d(y, tkernel(r, c), groups=c)


#### Bildverarbeitung und Augmentation ####

def downscale(imgs, f):
    b, w, h, c = imgs.shape
    imgs = imgs.reshape((b, w//f, f, h//f, f, c))
    imgs = imgs.mean(4).mean(2)
    return imgs

def fliprot(imgs):
    if np.random.rand()<0.5:
        imgs = imgs[:,::-1]
    if np.random.rand()<0.5:
        imgs = imgs[:,:,::-1]
    if np.random.rand()<0.5:
        imgs = imgs.transpose((0,2,1,3))
    return imgs

def fliprot_mult(imgs1, imgs2):
    if np.random.rand()<0.5:
        imgs1 = imgs1[:,::-1]
        imgs2 = imgs2[:,::-1]
    if np.random.rand()<0.5:
        imgs1 = imgs1[:,:,::-1]
        imgs2 = imgs2[:,:,::-1]
    if np.random.rand()<0.5:
        imgs1 = imgs1.transpose((0,2,1,3))
        imgs2 = imgs2.transpose((0,2,1,3))
    return imgs1, imgs2


def randomcrop_dual(y1, y2, h, w=None):
    if isinstance(w, type(None)):
        w = h
    if h<=1:
        h = int(h*y1.shape[1])
        w = int(w*y1.shape[2])
    dy = np.random.randint(y1.shape[1]-h)
    dx = np.random.randint(y1.shape[2]-w)
    y1 = y1[:, dy:dy+h, dx:dx+w, :]
    y2 = y2[:, dy:dy+h, dx:dx+w, :]
    return y1, y2



#############

relu = F.relu


def l1l2(x, y, r=1/10):
    # Mixture of l1 and l2 lossfunction
    return (x-y)**2 + r*torch.abs(x-y)


def L1L2Balanced(x, beta=1):
    'Balanced between l1 and l2 loss. Beta is the weight for l2. l1 is smoothed around 0.'
    m = torch.abs(x).mean()
    eps = m/10
    l1 = torch.sqrt(x**2+eps**2)-eps
    l2 = 1/2*x**2
    return l1+beta*l2

l1l2bal = lambda x, y: L1L2Balanced(x-y)


def doubleleaky(x, alpha=0.1):
    x = torch.nn.functional.leaky_relu(x, alpha)
    x = 1.-x
    x = torch.nn.functional.leaky_relu(x, alpha)
    x = 1.-x
    return x


def torch_randomcrop(y1, h, w=None):
    'Crops a single image (not a batch)'
    if isinstance(w, type(None)):
        w = h
    if h<=1:
        h = int(h*y1.shape[1])
        w = int(w*y1.shape[2])
    dy = torch.randint(y1.shape[1]-h, size=(1,))
    dx = torch.randint(y1.shape[2]-w, size=(1,))
    y1 = y1[:, dy:dy+h, dx:dx+w]
    return y1

def torch_randomcrop_batch(y1, y2, h, w=None):
    if isinstance(w, type(None)):
        w = h
    dy = torch.randint(y1.shape[2]-h, size=(1,))
    dx = torch.randint(y1.shape[3]-w, size=(1,))
    y1 = y1[:, :, dy:dy+h, dx:dx+w]
    y2 = y2[:, :, dy:dy+h, dx:dx+w]
    return y1, y2



def update_mt(mt, n, tau):
    # updates the mean teacher by the network
    mtdict = mt.state_dict()
    ndict = n.state_dict()
    for k in mtdict.keys():
        mtdict[k] = tau * mtdict[k] + (1-tau) * ndict[k]
    mt.load_state_dict(mtdict)



class AttentionLayer(nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.qkv = nn.Conv1d(n, 2*m + n, 1) # all in one layer: q and k have n//8 channels, v has n channels
        self.nm = (n,m)

        ##self.postlayer = nn.Conv2d(n, n, 1)

        self.gamma = nn.Parameter(torch.tensor([1.])) #### instead of 0.0

    def forward(self, x):
        x0 = x

        b, c, h, w = x.shape
        n, m = self.nm

        x = relu(x)

        qkv = self.qkv(x.view(b, c, h*w))
        #q, k, v = qkv[:,:c//2], qkv[:,c//2:2*c//2], qkv[:,2*c//2:]
        q, k, v = torch.split(qkv, [m, m, n], dim=1)
        beta = torch.bmm(q.permute(0,2,1), k) # has dimensions b, h*w, h*w
        beta = beta / np.sqrt(m)

        beta = F.softmax(beta, dim=1)
        ###self.last_beta = beta

        o = torch.bmm(v, beta)
        o = o.reshape(b, c, h, w)
        ##o = self.postlayer(o)

        return x0 + self.gamma * o


#=============================================================

avgpool = nn.AvgPool2d(2)

act = nn.ReLU()


def adjust_hw(x, h1, w1):
    zeros = lambda b, n, h, w: torch.zeros((b, n, h, w), device=x.device)
    b, n0, h0, w0 = x.shape
    if h1<h0 and w1<w0:
        rh = (h0-h1)//2
        sh = (h0-h1) - rh
        rw = (w0-w1)//2
        sw = (w0-w1) - rw
        # rh and sh add up to h1-h0
        x = x[:,:,rh:-sh,rw:-sw]
    elif h1>h0 and w1>w0:
        hr = (h1-h0)//2
        hs = (h1-h0) - rh
        wr = (w1-w0)//2
        ws = (w1-w0) - rw
        x = torch.cat((zeros(b, n0, hr, w0), x, zeros(b, n0, hs, w0)), dim=2)
        x = torch.cat((zeros(b, n0, h1, sr), x, zeros(b, n0, h1, ws)), dim=3)
    elif h1==h0 and w1==w0:
        pass
    else:
        raise NotImplemented # Höhe größer aber Weite geringer oder umgekehrt
    assert x.shape[2]==h1 and x.shape[3]==w1, (x.shape, h1, w1)
    return x

def adjust_n(x, n1):
    # Adjust channel number
    zeros = lambda b, n, h, w: torch.zeros((b, n, h, w), device=x.device)
    b, n0, h0, w0 = x.shape
    if n1 > n0:
        x = torch.cat((x, zeros(b, n1-n0, h0, w0)), dim=1)
    elif n1 < n0:
        x = n1/n0 * sum([ x[:,i:i+n1] for i in range(0,n0,n1)])
    assert x.shape[1] == n1, (x.shape, n1)
    return x

#def addskip(x, xskip):
#    xskip = adjust_n(xskip, x.shape[1])
#    return x+xskip






def expandtoeven(x):
    'Expands a tensor such that its multiple of two'
    b, c, h, w = x.shape
    if h%2==1:
        x = torch.nn.functional.pad(x, (0, 1, 0, 0))
    if w%2==1:
        x = torch.nn.functional.pad(x, (0, 0, 0, 1))
    return x


def addskip(x, xskip):
    'Adds x and skip connection, adjusts the shape to the skip connection'
    b, c, h, w = xskip.shape
    b, c, hp,wp= x.shape
    if hp!=h:
        r = (hp-h)//2
        s = hp-h - r
        # r and s add up to hp-h = wp-w
        x = x[:,:,r:-s,r:-s]
    return x+xskip







#  These are changed compared to hex_nnet version for 6x6 games!
class ResBlock(nn.Module):
    def __init__(self, n, n1=None, sz=3, bn=True):
        'Like a ResNet Block but without the residual yet added'
        super().__init__()
        if type(n1) is type(None): n1 = n
        nmid = int((n+n1)/2)
        self.bn = bn
        self.n1 = n1
        self.conv1 = nn.Conv2d(n, nmid, sz, padding=sz//2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(nmid, n1, sz, padding=sz//2, padding_mode='reflect')
        if bn:
            self.bn1 = nn.BatchNorm2d(nmid)
            self.bn2 = nn.BatchNorm2d(n1)

    def forward(self, x):
        x0 = x
        if self.bn:
            x = self.bn1(x)
            x = act(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = act(x)
            x = self.conv2(x)
        else:
            x = act(x)
            x = self.conv1(x)
            x = act(x)
            x = self.conv2(x)
        x0 = adjust_n(x0, x.shape[1])
        return x0 + x

'''
def adjustto(x, y):
    'Adjusts the shape of x to that of y'
    # Adjust height and width
    zeros = lambda b, n, h, w: torch.zeros((b, n, h, w), device=y.device)
    b, n0, h0, w0 = x.shape
    b, n1, h1, w1 = y.shape
    if h1<h0 and w1<w0:
        rh = (h0-h1)//2
        sh = (h0-h1) - rh
        rw = (w0-w1)//2
        sw = (w0-w1) - rw
        # rh and sh add up to h1-h0
        x = x[:,:,rh:-sh,rw:-sw]
    elif h1>h0 and w1>w0:
        hr = (h1-h0)//2
        hs = (h1-h0) - rh
        wr = (w1-w0)//2
        ws = (w1-w0) - rw
        x = torch.cat((zeros(b, n0, hr, w0), x, zeros(b, n0, hs, w0)), dim=2)
        x = torch.cat((zeros(b, n0, h1, sr), x, zeros(b, n0, h1, ws)), dim=3)
    elif h1==h0 and w1==w0:
        pass
    else:
        raise NotImplemented # Höhe größer aber Weite geringer oder umgekehrt

    # Adjust channel number
    if n1 > n0:
        x = torch.cat((x, zeros(b, n1-n0, h1, w1)), dim=1)
    elif n1 < n0:
        x = n1/n0 * sum([ x[:,i:i+n1] for i in range(0,n0,n1)])

    assert x.shape==y.shape, (x.shape, (b,n0,h0,w0), (b,n1,h1,w1))
    return x'''



class ResBlockDown(nn.Module):
    def __init__(self, nin, nout, ks=3, bn=True):
        'Downscaling Resnet Block'
        super().__init__()
        assert ks%2==1
        self.conv_res = nn.Conv2d(nin, nout, 1)
        self.conv1 = nn.Conv2d(nin, nout, ks, padding=ks//2, stride=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(nout, nout, ks, padding=ks//2, padding_mode='reflect')
        if bn:
            self.bn_res = nn.BatchNorm2d(nin)
            self.bn1 = nn.BatchNorm2d(nin)
            self.bn2 = nn.BatchNorm2d(nout)
        else:
            self.bn_res = lambda x: x
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x

    def forward(self, x):
        xr = avgpool(x)
        xr = self.conv_res(self.bn_res(xr))
        x  = self.conv1(act(self.bn1(x)))
        x  = self.conv2(act(self.bn2(x)))
        x  = xr + x
        return x


class ResBlockUp(nn.Module):
    def __init__(self, nin, nout):
        'Upscaling Resnet Block (actually not a Resnet Block)'
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
    def forward(self, x):
        x = act(self.conv(x))
        return x



def update_mt(mt, n, tau):
    # updates the mean teacher by the network
    mtdict = mt.state_dict()
    ndict = n.state_dict()
    for k in mtdict.keys():
        mtdict[k] = tau * mtdict[k] + (1-tau) * ndict[k]
    mt.load_state_dict(mtdict)


def softmax2d(x):
    b, c, h, w = x.shape
    x = x.reshape(b, -1)
    x = nn.functional.softmax(x, dim=-1)
    return x.reshape(b, c, h, w)



def pairplot(a, b):
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(np.array([a,b]).T)
    sns.pairplot(df)
