import copy
from typing import Optional, Any, Union, Callable

import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module, ModuleList, Dropout, Linear, LayerNorm
from torch.nn.modules import ModuleList
from torch.nn.init import xavier_uniform_
import torchvision.ops.boxes as box_ops

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def swish(x):
    return x * torch.sigmoid(x)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish,
          "sigmoid": F.sigmoid, "tanh": F.tanh, "softplus": F.softplus}

activation = "gelu"


class ScaledDotProductAttention(Module):

    def forward(self, query, key, value, sa, sv, mask=None):
        """
            q, k, v: bs * n_head, n_len, dim
        """
        bs_nh, nk, dk = query.size()# [-2]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        v_val = attention.matmul(value)
        
        if sa is not None and sv is not None:
            sa = sa.reshape(-1, nk, nk, dk)
            sv = sv.reshape(-1, nk, nk, dk)
            s_val = (sa * sv).mean(dim=2) 
            return torch.cat([v_val, s_val], dim=-1), attention 
        else:
            return v_val, attention


class MultiheadAttention(Module):

    def __init__(self,
                 in_features,
                 head_num,
                 dropout=0.1,
                 use_sp=False,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiheadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.use_sp = use_sp

        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.dropout = nn.Dropout(dropout)

        if self.use_sp:
            self.linear_sa = nn.Sequential(
                nn.Linear(36, in_features, bias),
                nn.BatchNorm1d(in_features),
                nn.Tanh(),
                nn.Linear(in_features, in_features, bias), 
                nn.BatchNorm1d(in_features),
                nn.Tanh(),
                nn.Linear(in_features, in_features, bias),
            )
            self.linear_sv = nn.Sequential(
                nn.Linear(36, in_features, bias),
                nn.BatchNorm1d(in_features),
                nn.Tanh(),
                nn.Linear(in_features, in_features, bias), 
                nn.BatchNorm1d(in_features),
                nn.Tanh(),
                nn.Linear(in_features, in_features, bias),                           
            )
            self.linear_o = nn.Linear(in_features * 2, in_features, bias)
        else:
            self.linear_o = nn.Linear(in_features, in_features, bias)


    def get_center(self, boxes):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        return (x1 + x2) / 2., (y1 + y2) / 2.


    def get_pairwise_spatial(self, b1, b2, eps=1e-8):
        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x, c1_y, c2_x, c2_y,
            # Relative box width and height
            b1_w, b1_h, b2_w, b2_h,
            # Relative box area
            b1_w * b1_h, b2_w * b2_h,
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)
        return torch.cat([f, torch.log(f + eps)], 1)

    def forward(self, q, k, v, dist, boxes, hi, oi, mask):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.use_sp:
            bh = boxes[hi]; bo = boxes[oi]
            visual_diff_feat = self.get_pairwise_spatial(bh, bo)
            sa = self.linear_sa(visual_diff_feat)
            sv = self.linear_sv(visual_diff_feat)
            sa = sa.unsqueeze(0)
            sv = sv.unsqueeze(0)

            if self.activation is not None:
                q = self.activation(q)
                k = self.activation(k)
                v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if self.use_sp:
            sa = self._reshape_to_batches(sa)
            sv = self._reshape_to_batches(sv)

        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)

        if self.use_sp:
            y, attn = ScaledDotProductAttention()(q, k, v, sa, sv, mask)
        else:
            y, attn = ScaledDotProductAttention()(q, k, v, None, None, mask)
        y = self._reshape_from_batches(y)
        
        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)

        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class TransformerEncoderLayer(Module):

    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout_prob=0.1, use_sp=False, 
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        self.use_sp = use_sp
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout_prob, use_sp=use_sp)
        self.linear1 = Linear(d_model, dim_feedforward) 
        self.dropout = Dropout(dropout_prob)
        self.linear2 = Linear(dim_feedforward, d_model) 

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps) 
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps) 
        self.dropout1 = Dropout(dropout_prob)
        self.dropout2 = Dropout(dropout_prob)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)


    def forward(self, src: Tensor, dist, boxes, hi, oi, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            output, attn = self._sa_block(self.norm1(x), dist, boxes, hi, oi, src_mask, src_key_padding_mask)
            x = x + output
            x = x + self._ff_block(self.norm2(x))
        else:
            output, attn = self._sa_block(x, dist, boxes, hi, oi, src_mask, src_key_padding_mask)
            x = self.norm1(x + output)
            x = self.norm2(x + self._ff_block(x))

        return x, attn


    # self-attention block
    def _sa_block(self, x: Tensor, dist: Tensor, boxes, hi: Tensor, oi: Tensor, attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn = self.self_attn(x, x, x, dist, boxes, \
                           hi, oi, attn_mask)
        return self.dropout1(x), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = ACT2FN[activation]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = ACT2FN[activation]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertBiAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob1=0.1, dropout_prob2=0.1, visualization=False, n_s=4):
        super(BertBiAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads)
            )

        self.visualization = visualization
        self.num_attention_heads = num_heads
        self.num_spatial_heads = min(n_s, self.num_attention_heads)
        self.attention_head_size = int(
            hidden_size / num_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(hidden_size, self.all_head_size)
        self.key1 = nn.Linear(hidden_size, self.all_head_size)
        self.value1 = nn.Linear(hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(dropout_prob1)

        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(dropout_prob2)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape) # bs, n, h, l
        return x.permute(0, 2, 1, 3) # bs, h, n, l

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        # n_h = input_tensor1.size()[1]
        # n_o = input_tensor2.size()[1]

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
 
        context_layer1 = torch.matmul(attention_probs1, value_layer1) # bs h n l
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous() # bs n d
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)
        # context_layer1 = self.dropout1(context_layer1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

 
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
   
        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)
        # context_layer2 = self.dropout2(context_layer2)

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data

class BertBiOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm1 = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.q_dense1 = nn.Linear(hidden_size, hidden_size)
        self.q_dropout1 = nn.Dropout(dropout_prob)

        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm2 = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.q_dense2 = nn.Linear(hidden_size, hidden_size)
        self.q_dropout2 = nn.Dropout(dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob0=0.1, dropout_prob1=0.1, dropout_prob2=0.1):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(hidden_size, num_heads, dropout_prob1, dropout_prob2)

        self.biOutput = BertBiOutput(hidden_size, dropout_prob0)

        self.v_intermediate = BertImageIntermediate(hidden_size)
        self.v_output = BertImageOutput(hidden_size, dropout_prob0)

        self.t_intermediate = BertIntermediate(hidden_size)
        self.t_output = BertOutput(hidden_size, dropout_prob0)


    def forward(
        self,
        input_tensor1,
        input_tensor2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            input_tensor2,
            co_attention_mask,
            use_co_attention_mask,
        )

        return bi_output2, bi_output1
