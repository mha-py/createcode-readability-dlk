


from layers import *


NBINS = 20

class Net(nn.Module):
    def __init__(self, n, nh, ntok, nbins=NBINS):
        super().__init__()
        
        self.emb = nn.Embedding(ntok, n)
        self.posenc = PositionalEncoding(n)
        
        self.ln1 = LayerNorm(n)
        self.enc1 = EncoderBlock(n, n, nh, bias=False)
        self.enc2 = EncoderBlock(n, n, nh, bias=False)
        self.enc3 = EncoderBlock(n, n, nh, bias=False)
        self.enc4 = EncoderBlock(n, n, nh, bias=False)
        self.ln2 = LayerNorm(n)
        self.dense1 = nn.Linear(n, n)
        self.dense2 = nn.Linear(n, nbins)
        
        self.dropout = nn.Dropout(0.1)
        self.cuda()
        self.d_model = n
        self.xlast = None
        self.x_enc_last = None
        
    def forward(self, x):
        x = self.emb(x) * np.sqrt(self.d_model)
        x = self.posenc(x)
        x = self.dropout(x)
        x = self.ln1(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.ln2(x)
        
        z = x[:,0,:]
        #z = torch.max(x, dim=1)[0]
        z = self.dropout(z)
        z = self.dense1(z)
        z = self.dropout(z)
        z = self.dense2(z)
        
        return z
