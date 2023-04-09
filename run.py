from torch import Tensor
from typing import Tuple
from model import TransformerModel
from data import Data
from config import Config
import torch
from tempfile import TemporaryDirectory
import os
import time
from train import Train
import math
from data import vocab

config = Config()
data = Data()
model = TransformerModel(len(vocab), config.emsize, config.nhead, config.d_hid, config.nlayers, config.dropout).to(config.device)
train = Train(model, data)

best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train.train()
        val_loss = train.evaluate()
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        train.scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states


test_loss = train.evaluate_test()
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)