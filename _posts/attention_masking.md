## Improving Attention Scoring by Muting Padded Tokens

### Introduction

The purpose of this report is to propose a methodology to improve the attention scoring in natural language processing (NLP) tasks by muting padded tokens. Padded tokens are used to standardize the length of sequences in a dataset. However, these padded tokens are not informative and should not affect the attention scoring. In this report, we propose two methodologies for muting the padded tokens, which are as follows:

- Masking attn score with Inf before softmax
- requires_grad=False for `<pad>` elements

The implementation details for the first methodology are discussed in this report, and the second methodology is briefly described.

### Methodology A: Masking attn score with Inf before softmax

In this methodology, we propose to mute the attention scores of padded tokens before the softmax function is applied. The following steps are proposed:

1. Create a mask with true values in place of real values and zeros in padded values.
2. Pass the mask to the attention class through the forward method of cronos high-level object.
3. Apply the mask to the attention scores before the softmax normalization using `TORCH.TENSOR.MASKED_FILL_`.

#### Mask characteristics

Masks are used to selectively update elements in a tensor. In this methodology, the mask is the same size as the tensor being masked, and only the elements with `True` values in the mask are updated.

#### Implementation details

To implement the proposed methodology, the following steps are suggested:

1. Pass the original length of each sequence to the model call.
2. Create a mask tensor input of `[batch size, source sentence length]` using the rules mentioned above.
3. Pass the mask to the attention class through the forward method of cronos high-level object.
4. Inside the attention class, apply the mask to the attention scores before the softmax normalization using `TORCH.TENSOR.MASKED_FILL_`.

The implementation of the proposed methodology may require refactoring of dataloaders to include sequence length, and the cronos preprocessing will need to be refactored to calculate the sequence length. Additionally, the fit, evaluate, and explain methods of the cronos API will need to be refactored to incorporate the proposed methodology.

### Methodology B: requires_grad=False for `<pad>` elements

In this methodology, we propose to set the `requires_grad` attribute to `False` for the padded tokens. By doing so, the gradients will not be calculated for the padded tokens during backpropagation.

### References

The following references were used to develop the proposed methodologies:

- https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_
- https://stats.stackexchange.com/questions/598239/how-is-padding-masking-considered-in-the-attention-head-of-a-transformer
- https://charon.me/posts/pytorch/pytorch_seq2seq_4/
- https://juditacs.github.io/2018/12/27/masked-attention.html


