import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + sorted(list({word for text in texts for word in text.lower().split()}))
        self.vocab_size = len(tokens)
        self.word_to_id = dict(zip(tokens, range(self.vocab_size)))
        self.id_to_word = dict(zip(range(self.vocab_size), tokens))

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        
        unknown_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(t, unknown_id) for t in text.lower().split()]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return ' '.join([self.id_to_word.get(id, self.unk_token) for id in ids])

            
