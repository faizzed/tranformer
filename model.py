import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from positional_encoding import PositionalEncoding
from data import vocab


def tensor_to_string(tensor: Tensor) -> str:
    for t in tensor:
      print( ", ".join(vocab.lookup_tokens(list(t))))


class TransformerModel(nn.Module):
    ntokens = 0
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.ntokens = ntoken
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embeddings = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embeddings(src) * math.sqrt(self.d_model) # we get the src as text which is batched but embeddings assign it dimension we know nothing about
        src = self.pos_encoder(src)
        # Its passed through multi head attention and then passed through feed forward network
        # mha init its own K, V, Q based on random weights and then use multi heads to learn the next possible word - this is all but just random numbers at this point.
        output = self.transformer_encoder(src, src_mask)
        # decoder is just a linear layer that takes the output of the transformer encoder and maps it to the vocab size
        output = self.decoder(output)
        return output