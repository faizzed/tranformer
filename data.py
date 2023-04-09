from torch.utils.data import IterableDataset
from torchtext.datasets import CoLA
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import Tensor
from torch import nn
from config import Config


def data_parser(data):
    return map(lambda x: x[2], data)


config = Config()
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, data_parser(CoLA(split='train'))), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
vocab.set_default_index(vocab['<unk>'])


class Data(nn.Module):
    train_data = None
    eval_data = None
    test_data = None
    vocab = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_iter, val_iter, test_iter = CoLA()

        self.train_data = self.data_to_batch(data_parser(train_iter))
        self.eval_data = self.data_to_batch(data_parser(val_iter))
        self.test_data = self.data_to_batch(data_parser(test_iter))

    def data_to_batch(self, data: IterableDataset) -> Tensor:
        processed = self.data_process(data)
        return self.batchify(processed, config.batch_size)

    def data_process(self, raw_text_iter: IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data: Tensor, bsz: int) -> Tensor:
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(config.device)
