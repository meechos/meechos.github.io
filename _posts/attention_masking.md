# Effective Attention Scoring for Padded Inputs

Commonly, attention layers will learn a weight for every token of a sequential input, so that $n_{attention_weights} = n_{input_tokens}$. Although it is reasonable to allow the model to learn how much each token in a sequence should contribute to model prediction, $Padding$ messes up this assumption. Let's find out.

## Sequence padding

Padding is a technique used in training neural networks for sequential inputs to ensure that all sequences have the same length. The reason for padding is to enable efficient processing of batches of sequences by the network. Without padding, sequences of different lengths would have to be processed separately, which is less computationally efficient.

The process of padding involves adding zeros (or some other value) to the end of a sequence to make it a fixed length. The length is typically chosen to be the length of the longest sequence in the dataset. During training, the network and its recurrent layers ignore the padded zeros, so the padding does not affect the output of the network. Padding is commonly used in natural language processing (NLP) tasks such as text classification, sentiment analysis, and machine translation.

## Padding conflicts with attention

Despite that the reccurent layers network ignores the padded zeros during training, when attentive layers are stacked on top of recurrency things get messy. 

![image](https://user-images.githubusercontent.com/429321/220680328-51beef43-7a14-4d72-a3bf-e37e0c7abdcb.png)

In specific the source of incorrecntness originates the fact that:
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

### Methodology B: requires_grad=False for `<pad>` elements

In this methodology, we propose to set the `requires_grad` attribute to `False` for the padded tokens. By doing so, the gradients will not be calculated for the padded tokens during backpropagation.

```python
x = torch.randn(2, 3)
pad = torch.zeros(2, 3)

### References

The following references were used to develop the proposed methodologies:

- https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
- https://stats.stackexchange.com/questions/598239/how-is-padding-masking-considered-in-the-attention-head-of-a-transformer
- https://charon.me/posts/pytorch/pytorch_seq2seq_4/
- https://juditacs.github.io/2018/12/27/masked-attention.html


