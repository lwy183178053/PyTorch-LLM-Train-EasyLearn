import json
from torch.utils.data import Dataset
import torch


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建输入文本
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        # 修改为使用 detach().clone()
        X = input_ids[:-1].detach().clone().to(torch.long)
        Y = input_ids[1:].detach().clone().to(torch.long)
        attention_mask = attention_mask[:-1].detach().clone().to(torch.long)
        return X, Y, attention_mask


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples


    def __getitem__(self, index):
        sample = self.samples[index]

        text = self.tokenizer.apply_chat_template(
            sample['conversations'],
            tokenize=False,
            add_generation_prompt=False
        )
        # 构建输入文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        # 修改为使用 detach().clone()
        X = input_ids[:-1].detach().clone().to(torch.long)
        Y = input_ids[1:].detach().clone().to(torch.long)
        attention_mask = attention_mask[:-1].detach().clone().to(torch.long)
        return X, Y, attention_mask



if __name__ == "__main__":
    pass
