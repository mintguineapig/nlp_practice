from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset, DataLoader
import omegaconf
from typing import List, Tuple, Literal

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, 
                 split : Literal['train', 'valid', 'test']):
        
        self.split = split
        self.max_len = data_config.max_len
        
        # Data loading & split
        imdb = load_dataset('imdb')
        combined = concatenate_datasets([imdb['train'], imdb['test']])
        train_val_test = combined.train_test_split(test_size=0.1, seed=42)
        train_data = train_val_test['train']
        val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=42)
        val_data = val_test['train']
        test_data = val_test['test']
        
        if split == 'train':
            raw_data = train_data
        elif split == 'valid':
            raw_data = val_data
        else: 
            raw_data = test_data

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.model_name)
        texts = [sample['text'] for sample in raw_data] 
        self.dataset = self.tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        labels = [sample['label'] for sample in raw_data]
        self.dataset['label'] = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.dataset['input_ids'])

    def __getitem__(self, idx: int) -> dict:
        return {key: self.dataset[key][idx] for key in self.dataset}

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        keys = batch[0].keys()
        return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
    
def get_dataloader(data_config : omegaconf.DictConfig, 
                   split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    dataset = IMDBDataset(data_config, split)
    dataloader = DataLoader(dataset, 
                            batch_size=data_config.batch_size, 
                            shuffle=(split=='train'), 
                            collate_fn=IMDBDataset.collate_fn)
    return dataloader