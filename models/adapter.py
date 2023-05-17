import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Base Adapter class.

    Handle common operations such as residual connection.
    """

    def __init__(self, residual=True):
        super().__init__()

        assert isinstance(residual, (bool, float))
        if isinstance(residual, bool):
            residual = 0.5 if residual else 0.
        if isinstance(residual, float):
            assert 0. <= residual <= 1.

        self.residual = residual

    def residual_add(self, in_feats, new_feats):
        """Perform residual connection."""
        assert isinstance(self.residual, float)
        return in_feats * self.residual + new_feats * (1. - self.residual)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError


class IdentityAdapter(Adapter):
    """Trivial Adapter that does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__(residual=False)

        # dummy parameter to record the dtype & device
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, feats, valid_masks):
        """feats: [B, num_views, C]."""
        return feats

    @property
    def dtype(self):
        return self.dummy.dtype


class TransformerAdapter(Adapter):
    """Transformer Adapter which is order-invariant."""

    def __init__(
        self,
        in_dim,
        d_model=256,
        num_heads=4,
        ffn_dim=256 * 4,
        norm_first=True,
        num_layers=2,
        residual=False,
    ):
        super().__init__(residual=residual)

        self.d_model = d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers)

        self.in_proj = nn.Linear(in_dim, d_model)
        self.out_proj = nn.Linear(d_model, in_dim)

    def forward(self, feats, valid_masks):
        """Inter-view interaction via Attention.

        Args:
            feats: [B, num_views, C]
            valid_masks: [B, num_views], True for valid views.
                Should mask the Attention in Transformer accordingly.
        """
        in_feats = feats

        # [B, num_views, d_model]
        feats = self.in_proj(feats)

        # [B, num_views, d_model]
        pad_masks = (~valid_masks)  # True --> padded
        feats = self.transformer_encoder(feats, src_key_padding_mask=pad_masks)

        # [B, num_views, C]
        feats = self.out_proj(feats)

        # residual connection
        feats = self.residual_add(in_feats, feats)

        return feats

    @property
    def dtype(self):
        return self.in_proj.weight.dtype
