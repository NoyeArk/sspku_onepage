"""
Dataset and Vocabulary for Code Generation (CONCODE)
"""
import json
import torch
from torch.utils.data import Dataset
from collections import Counter
import os


class Vocabulary:
    """Simple vocabulary wrapper for token-to-index mapping."""
    
    PAD_TOKEN = '<PAD>'
    BOS_TOKEN = '<BOS>'  # Begin of sequence
    EOS_TOKEN = '<EOS>'  # End of sequence
    UNK_TOKEN = '<UNK>'  # Unknown token
    
    def __init__(self):
        self.token2idx = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.n_tokens = 4
    
    def add_token(self, token):
        """Add a token to vocabulary."""
        if token not in self.token2idx:
            self.token2idx[token] = self.n_tokens
            self.idx2token[self.n_tokens] = token
            self.n_tokens += 1
    
    def add_sentence(self, tokens):
        """Add all tokens in a sentence."""
        for token in tokens:
            self.add_token(token)
    
    def __call__(self, token):
        """Convert token to index."""
        return self.token2idx.get(token, self.token2idx[self.UNK_TOKEN])
    
    def decode(self, indices):
        """Convert indices back to tokens."""
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, self.UNK_TOKEN)
            if token in [self.EOS_TOKEN, self.PAD_TOKEN]:
                break
            tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self):
        return self.n_tokens


def build_vocab(data_paths, min_freq=2):
    """
    Build source (NL) and target (Code) vocabularies from data files.
    
    Args:
        data_paths: List of paths to JSONL data files
        min_freq: Minimum frequency to include token in vocabulary
    
    Returns:
        src_vocab: Vocabulary for source (NL)
        tgt_vocab: Vocabulary for target (Code)
    """
    src_counter = Counter()
    tgt_counter = Counter()
    
    # Count tokens
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping...")
            continue
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                nl = data.get('nl', '')
                code = data.get('code', '')
                
                # Tokenize by space
                nl_tokens = nl.split()
                code_tokens = code.split()
                
                src_counter.update(nl_tokens)
                tgt_counter.update(code_tokens)
    
    # Build vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    # Add tokens with frequency >= min_freq
    for token, freq in src_counter.items():
        if freq >= min_freq:
            src_vocab.add_token(token)
    
    for token, freq in tgt_counter.items():
        if freq >= min_freq:
            tgt_vocab.add_token(token)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    return src_vocab, tgt_vocab


class ConcodeDataset(Dataset):
    """
    Dataset for CONCODE code generation task.
    
    Input format (JSONL):
        {"code": "void function ( ) { return value ; }", 
         "nl": "return the value"}
    
    Training format:
        Source: NL tokens
        Target: Code tokens with BOS and EOS
    """
    
    def __init__(self, data_path, src_vocab, tgt_vocab, max_len=100, mode='train'):
        """
        Args:
            data_path: Path to JSONL data file
            src_vocab: Source vocabulary (for NL)
            tgt_vocab: Target vocabulary (for Code)
            max_len: Maximum sequence length
            mode: 'train', 'dev', or 'test'
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.mode = mode
        
        self.data = []
        self._load_data(data_path)
    
    def _load_data(self, data_path):
        """Load data from JSONL file."""
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found")
            return
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.data.append(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            src: Source token indices (NL)
            tgt: Target token indices (Code with BOS/EOS)
            label: Target token indices for training (Code shifted)
        """
        item = self.data[idx]
        nl = item.get('nl', '')
        code = item.get('code', '')
        
        # Tokenize
        src_tokens = nl.split()[:self.max_len]
        tgt_tokens = code.split()[:self.max_len]
        
        # Convert to indices
        src = [self.src_vocab(t) for t in src_tokens]
        tgt = [self.tgt_vocab.BOS_TOKEN_IDX] + [self.tgt_vocab(t) for t in tgt_tokens]
        tgt = tgt[:self.max_len]
        
        # Add EOS for training
        if self.mode == 'train':
            tgt = tgt + [self.tgt_vocab.EOS_TOKEN_IDX]
        
        return torch.tensor(src), torch.tensor(tgt)
    
    def get_raw_code(self, idx):
        """Get raw code string for evaluation."""
        return self.data[idx].get('code', '')
    
    def get_raw_nl(self, idx):
        """Get raw NL string."""
        return self.data[idx].get('nl', '')


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Pads sequences to the same length within a batch.
    """
    srcs, tgts = zip(*batch)
    
    # Get max lengths
    src_len = max(s.size(0) for s in srcs)
    tgt_len = max(t.size(0) for t in tgts)
    
    # Pad sequences
    padded_srcs = []
    padded_tgts = []
    src_masks = []
    
    for src in srcs:
        pad_len = src_len - src.size(0)
        padded = torch.cat([src, torch.zeros(pad_len, dtype=torch.long)])
        padded_srcs.append(padded)
        # PyTorch Transformer: True = IGNORE position (padding), False = attend
        src_masks.append(torch.cat([torch.zeros(src.size(0)), torch.ones(pad_len)]).bool())
    
    for tgt in tgts:
        pad_len = tgt_len - tgt.size(0)
        padded = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
        padded_tgts.append(padded)
    
    return (torch.stack(padded_srcs),
            torch.stack(padded_tgts),
            torch.stack(src_masks))


# Add convenient token indices to Vocabulary class
Vocabulary.PAD_TOKEN_IDX = 0
Vocabulary.BOS_TOKEN_IDX = 1
Vocabulary.EOS_TOKEN_IDX = 2
Vocabulary.UNK_TOKEN_IDX = 3


if __name__ == '__main__':
    # Test vocabulary building
    data_path = '../data/train.jsonl'
    if os.path.exists(data_path):
        src_vocab, tgt_vocab = build_vocab([data_path], min_freq=2)
        print(f"Test passed: src_vocab={len(src_vocab)}, tgt_vocab={len(tgt_vocab)}")
    else:
        print(f"Test data not found at {data_path}")
