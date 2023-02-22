---
layout: page
title: Effective Attention Scoring for Padded Inputs
permalink: /Attention_masking
tags: Deep Learning
---

Commonly, attention layers will learn a weight for every token of a sequential input, so that $n_{attention-weights} = n_{input-tokens}$. Although it is reasonable to allow the model to learn how much each token in a sequence should contribute to model prediction, __padding__ messes up this assumption. Let's find out why.

## Sequence padding

Padding is a technique used in training neural networks for sequential inputs to ensure that all sequences have the same length. The reason for padding is to enable efficient processing of batches of sequences by the network. Without padding, sequences of different lengths would have to be processed separately, which is less computationally efficient.

The process of padding involves adding zeros (or some other value) to the end of a sequence to make it a fixed length. The length is typically chosen to be the length of the longest sequence in the dataset. During training, the network and its recurrent layers ignore the padded zeros, so the padding does not affect the output of the network. Padding is commonly used in natural language processing (NLP) tasks such as text classification, sentiment analysis, and machine translation.

## Padding conflicts with attention

Despite that the reccurent layers network ignores the padded zeros during training, when attentive layers are stacked on top of recurrency things get messy. 

![image](https://user-images.githubusercontent.com/429321/220701296-7a0d50e7-5540-41d6-a766-bb9ce347f895.png)

In specific the source of in-correctness stems from the fact that:
- Theoretically, attention should not be paid to tokens that have been arbitrarily appended, given that their role is enabling batch calculation during training.
- Technically, the sum of attention weights (per attention vector) should sum up to one $\sum{a_{ij}} = 1$. The probability $a_{ij}$ reflects the importance of the annotation $h_j$ with respect to the previous hidden state s_{i-1} in deciding the next state $s_i$ and generating $y_i$ [Bahdanau 2015]. 

In essence, this means that zeroing out the attention weights that link with padding symbols will break the assumption that the sum of weights add up to one.

## Methodologies for effective attention masking

I identify here two methodologies for muting attention weights that link with padded tokens.

1. Masking attention weights with minus infinity before the calculation of softmax
2. Freeze attention weights of padded tokens in the computation graph, so that they do not receive upadtes.

The implementation details for the first methodology are discussed in this report, and the second methodology is briefly described.

### 1. Masking attention weights with `-Inf`

The goal of this method is to mute attention weights for padded tokens before the softmax function is applied in the calculation of attention. A breakdown of required steps are outlined below:

1.	Create a mask with a `True` flag in place of real sequence tokens and a `False` flag in place of padded tokens.
2.	Using inverse masking, set the pad values’ attention weights to negative infinity and then call softmax in the attention calculation. 
3.	Ensure that attention weights sum to 1.

### 1.2 Implementation details

Masks are used to selectively update elements in a tensor. In this methodology, the mask is the same size as the tensor being masked, and only the elements with `True` values in the mask are updated during training.

Implementation steps:
1. Pass the original length of each sequence to the model call:
  ```python
  output = model((input, input_len)
  ```
2. Create a mask tensor of `[batch size, source sentence length]` using the rules mentioned above. The mask should either be available by the time the computational graph's call stack hits attention calculation.
4. Apply the mask to the attention weights before softmax normalization using `TORCH.TENSOR.MASKED_FILL_`.

```python
score_vector = torch.tensor([1, 2, 3, 4],dtype=torch.float)
score_masked = src_sequence.masked_fill(mask=torch.tensor([0, 0, 1, 1],dtype=torch.bool), value=-np.inf)

print(score_masked) # tensor([1., 2., -inf, -inf])
b = F.softmax(score_masked,dim=0)
print(b) # tensor([0.2689, 0.7311, 0.0000, 0.0000])
```
As seen in the example code above, the attention weights sum up to $1$. 

### 2: Freezing of Attention Weight Neurons 

In this methodology, the idea is to freeze the set of nodes in the computational graph that abide a certain set of characteristics. Particularly, we would like to freeze the neurons that respond to attention weights of padded tokens.

In a neural network, freezing a neuron or a layer means to fix its weights or parameters, effectively preventing them from being updated during the training process. This is typically done to preserve important features learned in the earlier stages of training, or to avoid overfitting on a small dataset how ever here the process could be leveraged to set a set of attention weights to zero and deprive those from changing their value during training.

### 2.1 Implementation details

In pytorch, users can freeze a layer using the designated `TORCH.TENSOR.REQUIRES_GRAD` flag.

```python
score_vector = torch.tensor([1, 2, 3, 4],dtype=torch.float, requires_grad=True)
```

However, here we would require __partial freezing__ of the `score_vector`. In order to achieve that, we could recreate the `score_vector` by concatenating or stacking a tensor that does receive updates (meaningfull attention weights) and a tensor that does not (padding attention weights).

```python
mport torch 

attn_real = torch.randn((2, 3), requires_grad=True)
print(attn_real)
attn_pad = torch.zeros((2, 2), requires_grad=False)
print(attn_pad)

>>>
tensor([[-1.6349, -0.8626,  0.3347],
        [-0.9115,  0.9382,  0.5463]], requires_grad=True)
tensor([[0., 0.],
        [0., 0.]])
        
score_vector = torch.cat((attn_real,attn_pad), 1)
print(score_vector)

>>>
tensor([[-1.6349, -0.8626,  0.3347,  0.0000,  0.0000],
        [-0.9115,  0.9382,  0.5463,  0.0000,  0.0000]], grad_fn=<CatBackward0>)
```

Calling `.is_leaf` on the `score_vector` would return `False` indicating that this object is created as a result of a concatenation operation that originates the two differently initialised vectors, with and without receiving updates respectively.

Although the example above is illustrative of the high-level idea, the methodology is practically impossible to apply to our use case, that is variable length of source inputs. Specifically, a variable length of source inputs means that the number of pad tokens appended per input would have to be different. Consequently, the mini-batch training would be deemed impossible given we would have to piece together a different computational graph per training example. Essentially, this methodology would defeat its ultimate purpose that is, batch training.

### References

The following references were used to develop the proposed methodologies:

- https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
- https://stats.stackexchange.com/questions/598239/how-is-padding-masking-considered-in-the-attention-head-of-a-transformer
- https://pytorch.org/docs/stable/generated/torch.cat.html
- https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html#torch.Tensor.is_leaf
- https://pytorch.org/docs/master/notes/autograd.html
- https://charon.me/posts/pytorch/pytorch_seq2seq_4/
- https://juditacs.github.io/2018/12/27/masked-attention.html


