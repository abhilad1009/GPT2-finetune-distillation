import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gzip
from typing import List


class MovieDataset(Dataset):  
    def __init__(self, tokenizer, movie_titles: List, max_len: int, dataset_type: str,max_seq_len: int=30) -> None:
        self.max_len = max_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id
        self.movies = movie_titles
        self.dataset_type = dataset_type
        self.result = []
        self.populate()


    def __len__(self) -> int:
        return len(self.result)


    def __getitem__(self, item: int) -> torch.Tensor:
        return self.result[item]
    
    def populate(self) -> None:
        for movie in self.movies:
            movie_words = movie.split()
            movie_len = len(movie_words)
            if movie_len > 1:
                prefix = f"<len> {movie_len-1} <word> {movie_words[0]} <text> "
                movie = (" ").join(movie_words[1:])
            else:
                prefix = f"<len> {movie_len} <word> movie <text> "
                movie = (" ").join(movie_words[:])

            encoded_prefix = self.tokenizer.encode(prefix)
            if self.dataset_type=="train":
                encoded_movie = self.tokenizer.encode(movie)
                if len(encoded_movie)>self.max_len:
                    encoded_movie = encoded_movie[:self.max_len]
                encoded_input = encoded_prefix + encoded_movie
                if len(encoded_input)>self.max_seq_len:
                    encoded_input = encoded_input[:self.max_seq_len-1]
                padded = encoded_input + [self.eos_id]*(self.max_seq_len-len(encoded_input))
            elif self.dataset_type=="test":
                padded = encoded_prefix
            # print(len(padded))
            self.result.append(torch.tensor(padded))
