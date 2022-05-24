# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
import torch

from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.data.audio.data_cfg import S2SDataConfig
from fairseq.models.speech_to_text import S2TConformerEncoder
from fairseq.models.speech_to_speech import (
    S2UTTransformerModel,
    s2ut_architecture_base as s2ut_transformer_architecture_base,
)
from fairseq.models.transformer import (
    Linear,
)


logger = logging.getLogger(__name__)


class S2SConformerEncoder(S2TConformerEncoder):
    """Based on S2T transformer encoder, with support
    to incorporate target speaker embedding."""

    def __init__(self, args):
        super().__init__(args)

        self.spk_emb_proj = None
        if args.target_speaker_embed:
            self.spk_emb_proj = Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )

    def forward(
        self, src_tokens, src_lengths, tgt_speaker=None, return_all_hiddens=False
    ):
        out = super().forward(src_tokens, src_lengths, return_all_hiddens)

        if self.spk_emb_proj:
            x = out["encoder_out"][0]
            seq_len, bsz, _ = x.size()
            tgt_speaker_emb = tgt_speaker.view(1, bsz, -1).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, tgt_speaker_emb], dim=2))
            out["encoder_out"][0] = x

        return out


@register_model("s2ut_conformer")
class S2UTConformerModel(S2UTTransformerModel):
    """
    Direct speech-to-speech translation model with S2T Conformer encoder + Transformer discrete unit decoder
    """

    @staticmethod
    def add_args(parser):
        S2UTTransformerModel.add_args(parser)
        parser.add_argument("--depthwise-conv-kernel-size", default=31)
        parser.add_argument(
            "--attn-type",
            default=None,
            help="If not specified uses fairseq MHA. Other valid option is espnet for using conformer",
        )
        parser.add_argument(
            "--pos-enc-type",
            default="abs",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )

    @classmethod
    def build_encoder(cls, args):
        print(args)
        data_cfg = S2SDataConfig(Path(args.data) / args.config_yaml)
        args.input_feat_per_channel = data_cfg.input_feat_per_channel
        args.input_channels = data_cfg.input_transformed_channels

        encoder = S2SConformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder


@register_model_architecture("s2ut_conformer", "s2ut_conformer")
def s2ut_base_architecture(args):
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    s2ut_transformer_architecture_base(args)
