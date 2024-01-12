


from layers import *


class Net(nn.Module):
    def __init__(self, n, nh, ntok):
        super().__init__()
        
        self.emb = nn.Embedding(ntok, n)
        self.posenc = PositionalEncoding(n)
        
        self.ln1 = LayerNorm(n)
        self.enc1 = EncoderBlock(n, n, nh, bias=True)
        self.enc2 = EncoderBlock(n, n, nh, bias=True)
        self.enc3 = EncoderBlock(n, n, nh, bias=True)
        self.enc4 = EncoderBlock(n, n, nh, bias=True)
        self.enc5 = EncoderBlock(n, n, nh, bias=True)
        self.enc6 = EncoderBlock(n, n, nh, bias=True)
        self.ln2 = LayerNorm(n)
        self.dense1 = nn.Linear(n, ntok)
        
        self.dropout = nn.Dropout(0.1)
        self.cuda()
        self.d_model = n
        
    def forward(self, x):
        mask = np2t(np.tri(x.shape[1])[None]).type(torch.float32).cuda()
        x = self.emb(x) * np.sqrt(self.d_model)
        #x = self.posenc(x)
        x = self.dropout(x)
        x = self.ln1(x)
        x = self.enc1(x, mask)
        x = self.enc2(x, mask)
        x = self.enc3(x, mask)
        x = self.enc4(x, mask)
        a = 0.5
        x = self.enc5(x, mask)# * a + x * (1-a)
        x = self.enc6(x, mask)# * a + x * (1-a)
        x = self.ln2(x)
        
        x = self.dense1(x)
        
        return x
