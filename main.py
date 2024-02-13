from framework import dataloader, model
from torch.utils.data import DataLoader
import pandas as pd
import torch
import gzip
import argparse
import os

import warnings

warnings.simplefilter('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",type=str, help="path to dataset")
    parser.add_argument("--train_size",type=int, help="number of data points to sample",default=20000)
    parser.add_argument("--test_size",type=int, help="number of data points to sample",default=1000)
    parser.add_argument("--max_len",type=int, help="max seq length",default=10)
    parser.add_argument("--epochs",type=int, help="epochs to run training",default=20)
    parser.add_argument("--task",type=str, help="train or test or both")
    parser.add_argument("--train_model_path",type=str, help="path to trained model",default=None)
    # Parse arguments
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with gzip.open(args.data_file, 'rb') as f:
        movie_df = pd.read_table(f, sep='\t', na_values=["\\N","nan"])

    if args.task=="train":
        print("Running training")
        gpt2 = model.GPT2Movie(device)

        movie_df_sampled = movie_df[movie_df['region']=="US"].sample(args.train_size)
        movie_titles = movie_df_sampled['title'].tolist()

        dataset = dataloader.MovieDataset(gpt2.tokenizer, movie_titles, args.max_len, dataset_type="train")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        gpt2.train(train_dataloader=dataloader, epochs=args.epochs)
        gpt2.save("model/")

    if args.task=="test":
        print("Running evaluation")
        gpt2 = model.GPT2Movie(device,args.train_model_path)

        movie_test = movie_df[movie_df['region']=="US"].sample(args.test_size)
        movie_test = movie_test['title'].tolist()

        test_dataset = dataloader.MovieDataset(gpt2.tokenizer, movie_test, args.max_len, dataset_type="test")
        
        gpt2.eval(test_dataset)

    elif args.task=="both":
        print("Running training and evaluation")
        gpt2 = model.GPT2Movie(device)

        movie_df_sampled = movie_df[movie_df['region']=="US"].sample(args.train_size)
        movie_titles = movie_df_sampled['title'].tolist()

        dataset = dataloader.MovieDataset(gpt2.tokenizer, movie_titles, args.max_len, dataset_type="train")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        gpt2.train(train_dataloader=dataloader, epochs=args.epochs)
        gpt2.save("model/")

        movie_test = movie_df[movie_df['region']=="US"].sample(args.test_size)
        movie_test = movie_test['title'].tolist()

        test_dataset = dataloader.MovieDataset(gpt2.tokenizer, movie_test, args.max_len, dataset_type="test")
        
        gpt2.eval(test_dataset)