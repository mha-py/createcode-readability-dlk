
from bpe import encode, decode
from collections import defaultdict

class Tokenizer:
    def __init__(self, abc=set(), bpe_code=dict()):
        for c in ['<SOS>', '<EOS>']:
            assert c not in abc
        self.abc = abc.copy()
        self.abc.add('<S>')  # token for beginning of a sequence
        self.abc.add('</S>') # token for end of a sequence
        self.abc.add('<U>')  # token for unknown
        
        self.t2i = { t: i for i, t in enumerate(self.abc) }  # char to token
        self.i2t = { i: t for i, t in enumerate(self.abc) }  # token to char
        self.NTOK = len(self.abc)
        
        self.bpe_code = bpe_code
    
    def tokenize(self, str, ints=True, encd=True):
        tokens = encode(str, self.bpe_code) if encd else str
        return [ self.t2i[t] for t in tokens ] if ints else tokens
        
    def detokenize(self, tokens, ints=True):
        if ints: tokens = [ self.i2t[t] for t in tokens ]
        ###print(tokens)
        str = decode(tokens)
        return str