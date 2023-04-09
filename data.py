from torch.utils.data import IterableDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import Tensor
from torch import nn
from config import Config


config = Config()


def data_parser(data):
    if config.TEST:
        return map(lambda x: x[2], data)
    return data


tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, data_parser(config.data('train'))), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
vocab.set_default_index(vocab['<unk>'])


class Data(nn.Module):
    train_data = None
    eval_data = None
    test_data = None
    vocab = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        train_iter, val_iter, test_iter = config.data()

        print('Train data')
        self.train_data = self.data_to_batch(data_parser(train_iter))
        print('Eval data')
        self.eval_data = self.data_to_batch(data_parser(val_iter))
        print('Test data')
        self.test_data = self.data_to_batch(data_parser(test_iter))

    def data_to_batch(self, data: IterableDataset) -> Tensor:
        processed = self.data_process(data)
        print("Data size: ", processed.size())
        return self.batchify(processed, config.batch_size)

    def data_process(self, raw_text_iter: IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = []
        for item in raw_text_iter:
            item = tokenizer(item)
            item = vocab(item)
            item = torch.tensor(item, dtype=torch.long)
            data.append(item)

        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data: Tensor, bsz: int) -> Tensor:
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        print("Batched size: ", data.size())
        return data.to(config.device)
