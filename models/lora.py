import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MultiheadAttention


def lora_w_init_(lora_down, lora_up, r):
    """Initialize the LoRA weights."""
    nn.init.normal_(lora_down, std=1. / r)
    nn.init.zeros_(lora_up)


class LoraInjectedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        r=4,
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )

        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)

        lora_w_init_(self.lora_down.weight, self.lora_up.weight, r)

    def forward(self, input):
        return (self.linear(input) + self.lora_up(self.lora_down(input)))


class LoraInjectedProj(nn.Module):
    """Apply LoRA on the projection head in MultiHeadAttention.

    The Q/K/V projection weight W is nn.Parameter(d_model, in_dim).
    We learn a lora_down [r, in_dim] and a lora_up [d_model, r].
    """

    def __init__(self, proj, r=4):
        super().__init__()

        d_model, in_dim = proj.shape
        if r > min(d_model, in_dim):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(d_model, in_dim)}"
            )

        self.d_model = d_model
        self.r = r
        self.proj = proj  # original projection weight, nn.Parameter
        self.proj.requires_grad = False  # freeze the original weight

        self.lora_down = nn.Parameter(torch.empty(r, in_dim))
        self.lora_up = nn.Parameter(torch.empty(d_model, r))

        lora_w_init_(self.lora_down, self.lora_up, r)

    def forward(self):
        """Return the LoRA updated weight."""
        return (self.proj + self.lora_up @ self.lora_down)


class LoraInjectedMergedProj(nn.Module):
    """Apply LoRA on the **merged** projection head in MultiHeadAttention.

    The merged projection weight W is nn.Parameter(3*d_model, in_dim).
    We learn three (lora_down [r, in_dim] and lora_up [d_model, r]).
    """

    def __init__(self, merged_proj, r=4):
        super().__init__()

        d_model_3, in_dim = merged_proj.shape
        assert d_model_3 % 3 == 0, "MergedProj's dim must be divisible by 3"
        d_model = d_model_3 // 3
        if r > min(d_model, in_dim):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(d_model, in_dim)}"
            )

        self.d_model = d_model
        self.r = r
        self.merged_proj = merged_proj  # original projection weight
        self.merged_proj.requires_grad = False  # freeze the original weight

        self.lora_down_q = nn.Parameter(torch.empty(r, in_dim))
        self.lora_up_q = nn.Parameter(torch.empty(d_model, r))
        self.lora_down_k = nn.Parameter(torch.empty(r, in_dim))
        self.lora_up_k = nn.Parameter(torch.empty(d_model, r))
        self.lora_down_v = nn.Parameter(torch.empty(r, in_dim))
        self.lora_up_v = nn.Parameter(torch.empty(d_model, r))

        lora_w_init_(self.lora_down_q, self.lora_up_q, r)
        lora_w_init_(self.lora_down_k, self.lora_up_k, r)
        lora_w_init_(self.lora_down_v, self.lora_up_v, r)

    def forward(self):
        """Return the LoRA updated weight."""
        return torch.cat([
            self.merged_proj[:self.d_model] +
            self.lora_up_q @ self.lora_down_q,
            self.merged_proj[self.d_model:2 * self.d_model] +
            self.lora_up_k @ self.lora_down_k,
            self.merged_proj[2 * self.d_model:] +
            self.lora_up_v @ self.lora_down_v,
        ],
                         dim=0)


class LoraInjectedMHA(MultiheadAttention):
    """MultiHeadAttention with LoRA fine-tuning."""

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        is_batched = query.dim() == 3
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None
                                  or attn_mask is not None):
            why_not_fast_path = "key_padding_mask and attn_mask are not supported with NestedTensor input"
        elif not query.is_nested and key_padding_mask is not None and attn_mask is not None:
            why_not_fast_path = "key_padding_mask and attn_mask were both supplied"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight(),
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x.is_cuda or 'cpu' in str(x.device))
                          for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(
                [x.requires_grad for x in tensor_args]):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight(), self.in_proj_bias,
                    self.out_proj.weight, self.out_proj.bias, key_padding_mask
                    if key_padding_mask is not None else attn_mask,
                    need_weights, average_attn_weights)
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [
                    x.transpose(1, 0) for x in (query, key, value)
                ]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight(),
                k_proj_weight=self.k_proj_weight(),
                v_proj_weight=self.v_proj_weight(),
                average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight(),
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def build_lora_proj(proj, r):
    """Given an old projection, build a LoRA projection."""
    lora_proj = LoraInjectedProj(proj=proj, r=r)
    return lora_proj


def build_lora_merged_proj(merged_proj, r):
    """Given an old merged projection, build a LoRA merged projection."""
    lora_merged_proj = LoraInjectedMergedProj(merged_proj=merged_proj, r=r)
    return lora_merged_proj


def build_lora_mha(mha, r):
    """Given an old MHA, build a LoRA-MHA."""
    lora_mha = LoraInjectedMHA(
        embed_dim=mha.embed_dim,
        num_heads=mha.num_heads,
        dropout=mha.dropout,
        kdim=mha.kdim,
        vdim=mha.vdim,
        batch_first=mha.batch_first,
        device=mha.out_proj.weight.device,
        dtype=mha.out_proj.weight.dtype,
    )
    lora_mha.load_state_dict(mha.state_dict())
    # inject LoRA to projection head weights
    if mha._qkv_same_embed_dim:  # replace `in_proj_weight`
        lora_mha.in_proj_weight = build_lora_merged_proj(mha.in_proj_weight, r)
    else:  # replace `q_proj_weight`, `k_proj_weight`, `v_proj_weight`
        lora_mha.q_proj_weight = build_lora_proj(mha.q_proj_weight, r)
        lora_mha.k_proj_weight = build_lora_proj(mha.k_proj_weight, r)
        lora_mha.v_proj_weight = build_lora_proj(mha.v_proj_weight, r)
    # TODO: replace `out_proj.weight`
    return lora_mha


def inject_trainable_lora(model, r=4):
    """Replace all the MHA in `model` with LoRA-MHA."""
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            lora_mha = build_lora_mha(module, r)
            setattr(model, name, lora_mha)
    return model
