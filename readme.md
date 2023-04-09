# Rough understanding of a transformer "Attention is all you need"

Input sentence: "I love dogs."

Tokenize the sentence into words:

```
["I", "love", "dogs"]
```
Convert each word into an embedding of dimensions 3 using a pre-trained word embedding matrix:
```
I_emb = [0.1, 0.3, 0.5]
love_emb = [-0.2, 0.4, 0.1]
dogs_emb = [0.3, -0.1, 0.8]
```

Create the positional encoding for each position in the sentence. Let's use a simplified version of the sine and cosine functions to generate positional encoding vectors:

```
pos_1 = [sin(1), cos(1), sin(1)] = [0.84, 0.54, 0.84]
pos_2 = [sin(2), cos(2), sin(2)] = [0.91, -0.42, 0.91]
pos_3 = [sin(3), cos(3), sin(3)] = [0.14, -0.99, 0.14]
```

Add positional encoding to the word embeddings:

```
I_emb_with_pos = [0.1 + 0.84, 0.3 + 0.54, 0.5 + 0.84] = [0.94, 0.84, 1.34]
love_emb_with_pos = [-0.2 + 0.91, 0.4 - 0.42, 0.1 + 0.91] = [0.71, -0.02, 1.01]
dogs_emb_with_pos = [0.3 + 0.14, -0.1 - 0.99, 0.8 + 0.14] = [0.44, -1.09, 0.94]
```

**Pass the above three tensors through transformer encoder:** 

This is a stack of (n) identical encoder layers.

**Encoder Layer**:

Put simply (there could be more nuances) this has a multi-head self-attention mechanism, and a simple feed-forward layer.

**Multi-head self-attention mechanism:**

In multi-head attention, the input sequence is processed through multiple attention heads, each of which performs self-attention independently. Each attention head has its own set of learnable parameters (Query, Key, and Value matrices) that help it specialize in capturing different types of information from the input sequence.
These tensors are initialized randomly by multiplying the embeddings with their weights. Since the weights are adjusted during training these values change. When Q, K, V change they capture better relations, context and meaning of the sentence.

**How to calculate Q, K, V?**

```
Attention score of each word (V) = softmax(scale(Q * K))
Compute context vector (C) = Sigma(Attention score * V)
Combine context vectors from all heads (C) = Concat(C1, C2, C3, C4)
```
Pass it through a feed-forward layer (FFN) to get the output (O) = FFN(C) - this chooses the best possible word from the context vector.
