from transformers import AutoTokenizer
import torch
from .constants import MODEL_CHECK_POINT, MODEL_MAX_LENGTH
from typing import List, Dict


class DistilBertDataset:
    def __init__(self, post: List[str], label: List[int]):
        self.post = post
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECK_POINT)

    def __len__(self):
        return len(self.post)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not isinstance(self.post[idx], str):
            post = str(self.post[idx])
        post = self.post[idx]
        encoded_text = self.tokenizer.encode_plus(
            post,
            text_pair=None,
            add_special_tokens=True,
            max_length=MODEL_MAX_LENGTH,
            padding='max_length',
            truncation=True,
        )
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
            }
