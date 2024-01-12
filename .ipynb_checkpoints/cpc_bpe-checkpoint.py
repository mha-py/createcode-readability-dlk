# byte pair encoding
# Credits to http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html 
# and https://ufal.mff.cuni.cz/~helcl/courses/npfl116/ipython/byte_pair_encoding.html

from collections import Counter
from tqdm.notebook import tqdm, trange
import re



def explode(word):
    r = ''
    for c in word:
        r += c + ' '
    return r ##+ '</w>' ## cpc-Modus


def get_pair_stats(vocab):
    """Get counts of pairs of consecutive symbols."""

    pairs = {}
    for word, frequency in vocab.items():
        symbols = word.split()

        # count occurrences of pairs
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            current_frequency = pairs.get(pair, 0)
            pairs[pair] = current_frequency + frequency

    return pairs
    
def merge_vocab(pair, v_in):
    'Merges a pair in each word of the vocabular'
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as a tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


from itertools import chain

def create_code(text, num_merges, nmin=1, bpe_codes=None):

    # use all characters in `splitter` to split the text corpus into words
    # https://stackoverflow.com/questions/1059559/split-strings-into-words-with-multiple-word-boundary-delimiters
    words = re.split('\W+', text) # splits into words, ignoring the ' ', '.', ')' and so on

    vocab = Counter(words)
    vocab = { explode(w): n for w, n in vocab.items() if n > nmin }


    # we store the best pair during each iteration for encoding new vocabulary, more on this later
    if not bpe_codes: bpe_codes = {}
    for i in trange(num_merges):
        #print('\niteration', i)
        pair_stats = get_pair_stats(vocab)
        if not pair_stats:
            break
    
        best_pair = max(pair_stats, key=pair_stats.get)
        bpe_codes[best_pair] = i
    
        ##print('best pair:', best_pair)
        ##print(list(vocab)[14])
        vocab = merge_vocab(best_pair, vocab)
    
    #print('\nfinal vocabulary: ', vocab)
    #print('\nbyte pair encoding: ', bpe_codes)

    return bpe_codes


def encode_word(orig, bpe_codes):
    """Encode word based on list of BPE merge operations, which are applied consecutively"""

    word = tuple(orig) + ('</w>',)
    ##print("__word split into characters:__ <tt>{}</tt>".format(word))

    pairs = get_pairs(word)

    if not pairs:
        return orig

    iteration = 0
    while True:
        iteration += 1
        ##print("__Iteration {}:__".format(iteration))
        
        ##print("bigrams in the word: {}".format(pairs))
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        ##print("candidate for merging: {}".format(bigram))
        if bigram not in bpe_codes:
            ##print("__Candidate not in BPE merges, algorithm stops.__")
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        ##print("word after merging: {}".format(word))
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)
   
    return word
    

def encode(text, bpe_codes, verbose=False):
    # Encodes a whole text rather than a single word
    tokens = [ list(encode_word(w, bpe_codes)) for w in text.split(' ') ]   # hat das format [ ['w', 'as'], ['i', 'st'], ... ]
    concat = []
    if verbose: tokens = tqdm(tokens)
    for word in tokens: # word hat format ['w', 'as']
        concat += word + ['</w>']
    return concat

def decode(text):
    return ''.join(text).replace('</w>', ' ')

    