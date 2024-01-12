
from collections import defaultdict

class Tokenizer:
    def __init__(self, abc=set()):
        for c in ['{', '}', '^', '°']:
            assert c not in abc
        ##self.abc = set(list(''.join([self.prep(c) for c in abc])))
        ##self.abc.add('{') # token for beginning of a sequence
        ##self.abc.add('}') # token for end of a sequence
        
        #self.c2t = { c: t for t, c in enumerate(self.abc) }  # char to token
        #self.t2c = { t: c for t, c in enumerate(self.abc) }  # token to char
        self.c2t = defaultdict(lambda: 0, { chr(t) : t for t in range(256) })
        self.t2c = { t : chr(t) for t in range(256) }
        self.NTOK = 256
    
    def tokenize(self, str):
        return [ self.c2t[c] for c in str ]
        
    def detokenize(self, tokens):
        return ''.join([ self.t2c[t] for t in tokens ])

    