from model import TransformerModel
from data import Data
from config import Config
import torch
import os
from train import Train
from data import vocab
from train import generate_square_subsequent_mask
from sentences import sentences

config = Config()


class Generator:
    model: TransformerModel
    data: Data
    train: Train

    def __init__(self):
        self.model = TransformerModel(len(vocab), config.emsize, config.nhead, config.d_hid, config.nlayers, config.dropout).to(config.device)
        if self.should_train():
            self.init_rest()

    def should_train(self):
        return not os.path.exists(os.path.join(config.checkpoint_dir, 'best_model_params.pt'))

    def init_rest(self):
        self.data = Data()
        self.train = Train(self.model, self.data)
        self.train_model()

    def train_model(self):
        if not os.path.exists(os.path.join(config.checkpoint_dir, 'best_model_params.pt')):
            print('No checkpoint found -- best thing is to train the model first')
            print('Kicking off training - Ctrl+C to stop')
            self.train.run_trainings()
        return

    def generate(self, sentences):
        # if we have a checkpoint, load it
        self.model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model_params.pt')))
        self.model.eval()
        print('Loaded model from checkpoint')
        outs = []
        for sentence in sentences:
            out = self.generate_one(self.model, sentence)
            outs.append({
                'input': sentence,
                'output': out
            })

        return outs

    def generate_one(self, model, sentence):
        sentence = torch.tensor(vocab.lookup_indices(sentence.lower().split())).unsqueeze(1).to(config.device)
        prediction = model(sentence, generate_square_subsequent_mask(sentence.size(0)).to(config.device))
        # convert prediction to words
        return ' '.join([vocab.vocab.itos_[t] for t in prediction.argmax(dim=2).squeeze().tolist()])


model = Generator()
out = model.generate(sentences)
for o in out:
    print(f'{o["input"]} >> {o["output"]}')