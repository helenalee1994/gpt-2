import glob
import numpy as np
import os
import tensorflow as tf
import tqdm
import random
from .save import load_pickle

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
        if 'chunk' in path:
            return load_pickle(path)
        
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
                tokens = enc.encode(raw_text) #+ '<|endoftext|>') 
                token_chunks.append(tokens)
                
    # list of list
    return token_chunks 


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, 
                 chunks, 
                 shuffle_ingredients=True, 
                 shuffle_fields=True, 
                 seed=None,
                 max_ingred = None, 
                 max_token = 512):
        
        self.chunks = chunks #[recipe for recipe in chunks if len(recipe)<= max_token]
        self.n_documents = len(self.chunks)
        self.rs = np.random.RandomState(seed=seed)
        self.seed = seed
        self.shuffle_ingredients = shuffle_ingredients
        self.shuffle_fields = shuffle_fields
        self.max_ingred = max_ingred
        
        # shuffle-related
        self.targets = {' <start-directions>': [1279, 9688, 12, 12942, 507, 29],
                        ' <start-ingredients>': [1279, 9688, 12, 278, 23320, 29],
                        ' <start-title>': [1279, 9688, 12, 7839, 29]}
        self.end_tag = [1279, 437, 12, 278, 23320, 29]
        
    def sample(self, length): #, shuffle_ingredients = True, shuffle = True):
        while True:
            index = self.rs.randint(0, self.n_documents)
            tokens = self.chunks[index]
            if self.shuffle_fields:
                tokens = self.shuffle(tokens)
            elif self.shuffle_ingredients:
                tokens = self.shuffle(tokens, ingred_only = True)
                
            # BPE encoding for '<|endoftext|>'
            tokens += [27, 91, 437, 1659, 5239, 91, 29]
            # import pdb; pdb.set_trace()
            # if not no constraints
            if not length == 0:
                diff = length - len(tokens)
                if diff > 0:
                    tokens += [16791] * diff # 16791 corresponds to <<
                elif diff < 0:
                    start = self.rs.randint(0, abs(diff))
                    tokens = tokens[start:start+length]
            
            return np.array(tokens)

    def shuff_ingredients(self, encoded_file):
        random.seed(self.seed)
        start, end, output = len(self.targets[' <start-ingredients>'])+1, 0, []
        for idx, token in enumerate(encoded_file):
            if idx >= start and encoded_file[idx] in [3]:
                end = idx+1
                output.append(encoded_file[start:end])
                start = idx+1
        random.shuffle(output)
        if self.max_ingred:
            output = output[:self.max_ingred]
            
        return  self.targets[' <start-ingredients>'] +sum(output, []) + self.end_tag

    def shuffle(self, encoded_file, ingred_only = False):
        ''' main version
        Args: encoded_file: a list encodes e.g. ' <start-title>easy, crunchy hot dogs <end-title> <start-ingr...'
        '''
        random.seed(self.seed)
        idx_targets = {}
        # read list
        start, end, output = 0, 0, []
        for idx, token in enumerate(encoded_file):
            if encoded_file[idx: idx+2] ==[1279, 9688]:
                end = idx
                if start != 0 and self.shuffle_ingredients:
                    field = encoded_file[start-1:end]
                    field = self.shuff_ingredients(field)
                else:
                    field = encoded_file[start:end]
                output.append(field)
                start = idx
        output.append(encoded_file[start:])
        
        if not ingred_only:
            # shuffle each fields
            random.shuffle(output)
            
        return sum(output, [])