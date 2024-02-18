import linecache
import json
import numpy as np
from torch.utils.data import Dataset


class InferenceDataset(Dataset):
    def __init__(self, filename, tokenizer, max_input_length):
        super(InferenceDataset, self).__init__()
        self._filename = filename
        self._tokenizer = tokenizer
        self._max_input_length = max_input_length
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        sample = json.loads(line)

        source = sample["prompt"]
        source_encode = self._tokenizer(source, padding="max_length", max_length=self._max_input_length, truncation=True)

        batch = {
            "input_ids": np.asarray(source_encode.input_ids),
            "attention_mask": np.asarray(source_encode.attention_mask),
            "input": sample["prompt"],
            "label": sample["completion"]
        }

        return batch

    def __len__(self):
        return self._total_data