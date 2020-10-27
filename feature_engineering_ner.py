# -*- encoding: utf-8 -*-
'''
@File    :   feature_engineering_ner.py
@Time    :   2020/10/26 20:53:37
@Author  :   Luo Jianhui 
@Version :   1.0
@Contact :   kid1412ljh@outlook.com
'''

# here put the import lib
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tag_list = ['O', 'B-影像检查', 'I-影像检查', 'B-解剖部位', 'I-解剖部位', 'B-疾病和诊断', 'I-疾病和诊断', 'B-药物', 'I-药物', 'B-实验室检验', 'I-实验室检验', 'B-手术', 'I-手术']

def all_label(path):
    clean_data = pd.read_csv(path)
    labels = clean_data['label']
    tag_dict = {}
    for i, c in enumerate(tag_list):
        tag_dict[c] = i + 1
    all_label = []
    for l in tqdm(labels):
        label_single = []
        for c in l.split():
            label_single.append(tag_dict[c])
        if len(label_single) > 510:
            label_single = label_single[0:510]
        else:
            label_single.extend([0 for _ in range(510-len(label_single))])
        label_single.insert(0,0)
        label_single.append(0)
        all_label.append(label_single)

    return all_label


def tokenize(path):
    """Tokenize all of the sentences and map the tokens to thier word IDs.

    """
    data = pd.read_csv(path)
    sentences = data.text.values
    labels = all_label(path)

    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #    (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode. 
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, attention_masks, labels

def split_data(path):
    
    input_ids, attention_masks, labels = tokenize(path)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

if __name__ == '__main__':
    path = 'total_data.csv'
    train, val = split_data(path)

