import glob
import numpy as np
import os
import tensorflow as tf
import tqdm

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text = fp.read()
                tokens = enc.encode(raw_text+ '<|endoftext|>') 
                token_chunks.append(tokens)
                
    # list of list
    return token_chunks 


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.n_documents = len(chunks)
        self.rs = np.random.RandomState(seed=seed)
    def sample(self, length):
        while True:
            index = self.rs.randint(0, self.n_documents)
            tokens = self.chunks[index]
            
            # if not no constraints
            if not length == 0:
                diff = length - len(tokens)
                if diff > 0:
                    tokens += [16791] * diff # 16791 corresponding to <<
                else:
                    tokens = tokens[:length]
            return np.array(tokens)
