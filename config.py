import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    eval_batch_size = 10
    emsize = 4  # embedding dimension
    d_hid = 4  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 1  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    bptt = 35 # batch size