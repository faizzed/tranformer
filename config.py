import torch
from torchtext.datasets import CoLA
from torchtext.datasets import WikiText2


class Config:
    device = None
    batch_size = None
    eval_batch_size = None
    emsize = None  # embedding dimension
    d_hid = None  # dimension of the feedforward network model in nn.TransformerEncoder (e.g number of neurons in a layer)
    nlayers = None  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = None  # number of heads in nn.MultiheadAttention
    dropout = None  # dropout probability
    bptt = None # batch size
    epochs = None  # The number of epochs
    checkpoint_dir = './checkpoints'
    TEST = False

    def __init__(self):
        if self.TEST:
            self.test_setting()
        else:
            self.normal_setting()

    def test_setting(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 20
        self.eval_batch_size = 10
        self.emsize = 4
        self.d_hid = 2
        self.nlayers = 2
        self.nhead = 1
        self.dropout = 0.2
        self.bptt = 35
        self.epochs = 2

    def normal_setting(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 20
        self.eval_batch_size = 10
        self.emsize = 200
        self.d_hid = 200
        self.nlayers = 2
        self.nhead = 5
        self.dropout = 0.2
        self.bptt = 35
        self.epochs = 3

    def data(self, split = 'none'):
        if self.TEST:
            if split == 'train':
                return CoLA(split='train')
            return CoLA()
        else:
            if split == 'train':
                return WikiText2(split='train')
            return WikiText2()

