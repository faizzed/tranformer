import copy
import time
from torch import nn
import torch
from torch import Tensor
from config import Config
from data import Data
from typing import Tuple
import math

config = Config()


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(config.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

class Train():
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    model = None
    optimizer = None
    scheduler = None
    data = None

    def __init__(self, model, data: Data):
        self.model = model
        self.data = data
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def train(self) -> None:
        self.model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(config.bptt).to(config.device)

        num_batches = len(self.data.train_data) // config.bptt
        for batch, i in enumerate(range(0, self.data.train_data.size(0) - 1, config.bptt)):
            data, targets = get_batch(self.data.train_data, i)
            seq_len = data.size(0)
            if seq_len != config.bptt:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = self.model(data, src_mask)
            loss = self.criterion(output.view(-1, self.model.ntokens), targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {i:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

    def evaluate(self) -> float:
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(config.bptt).to(config.device)
        with torch.no_grad():
            for i in range(0, self.data.eval_data.size(0) - 1, config.bptt):
                data, targets = get_batch(self.data.eval_data, i)
                seq_len = data.size(0)
                if seq_len != config.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask)
                output_flat = output.view(-1, self.model.ntokens)
                total_loss += seq_len * self.criterion(output_flat, targets).item()
        return total_loss / (len(self.data.eval_data) - 1)

    def evaluate_test(self) -> float:
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(config.bptt).to(config.device)
        with torch.no_grad():
            for i in range(0, self.data.test_data.size(0) - 1, config.bptt):
                data, targets = get_batch(self.data.test_data, i)
                seq_len = data.size(0)
                if seq_len != config.bptt:
                    src_mask = src_mask[:seq_len, :seq_len]
                output = self.model(data, src_mask)
                output_flat = output.view(-1, self.model.ntokens)
                total_loss += seq_len * self.criterion(output_flat, targets).item()
        return total_loss / (len(self.data.test_data) - 1)
