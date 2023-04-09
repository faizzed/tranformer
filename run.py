from model import TransformerModel
from data import Data
from config import Config
import torch
import os
from train import Train
from data import vocab
from train import generate_square_subsequent_mask

config = Config()


class Generator:
    model: TransformerModel
    data: Data
    train: Train

    def __init__(self):
        self.data = Data()
        self.model = TransformerModel(len(vocab), config.emsize, config.nhead, config.d_hid, config.nlayers, config.dropout).to(config.device)
        self.train = Train(self.model, self.data)
        self.train_model()

    def train_model(self):
        if not os.path.exists(os.path.join(config.checkpoint_dir, 'best_model_params.pt')):
            print('no checkpoint found -- best thing is to train the model first')
            print('kicking off training - Ctrl+C to stop')
            self.train.run_trainings()
        return

    def generate(self, sentence):
        # if we have a checkpoint, load it
        self.model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model_params.pt')))
        self.model.eval()
        print('loaded model from checkpoint')
        sentence = torch.tensor(vocab.lookup_indices(sentence.lower().split())).unsqueeze(1).to(config.device)
        prediction = self.model(sentence, generate_square_subsequent_mask(sentence.size(0)).to(config.device))
        # convert prediction to words
        return ' '.join([vocab.vocab.itos_[t] for t in prediction.argmax(dim=2).squeeze().tolist()])


model = Generator()
out = model.generate('Whats the capital of Germany')
print(out)