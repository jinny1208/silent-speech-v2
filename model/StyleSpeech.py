import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    MelStyleEncoder,
    # PhonemeEncoder,
    PhonemeEncoderWoutStyle,
    EmgEncoder,
    EMG2PhonemeAligner,
    # MelDecoder,
    MelDecoderWoutStyle,
    VarianceAdaptor,
    PhonemeDiscriminator,
    StyleDiscriminator,
)
from utils.tools import get_mask_from_lengths


class StyleSpeech(nn.Module):
    """ StyleSpeech """

    def __init__(self, preprocess_config, model_config):
        super(StyleSpeech, self).__init__()
        self.model_config = model_config

        self.phoneme_encoder_WoutStyle = PhonemeEncoderWoutStyle(model_config)
        self.emg_encoder = EmgEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.mel_decoder_WoutStyle = MelDecoderWoutStyle(model_config)
        self.phoneme_linear = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_hidden"],
        )
        self.emg_aligner = EMG2PhonemeAligner(d_model=model_config["EMGTransformer"]["encoder_hidden"], use_mha_heads=4)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.D_t = PhonemeDiscriminator(preprocess_config, model_config)
        self.D_s = StyleDiscriminator(preprocess_config, model_config)

        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            ),
            "r",
        ) as f:
            n_speaker = len(json.load(f))
        self.style_prototype = nn.Embedding(
            n_speaker,
            model_config["melencoder"]["encoder_hidden"],
        )

    def forward(
        self,
        _,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        emg=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        output = self.phoneme_encoder_WoutStyle(texts, src_masks) # 16, 171, 256 --> phoneme_encoder_WoutStyle로 바꿨을 때의 shape: 16, X, 256
        if emg is not None:
            emg_output = self.emg_encoder(emg) # emg_output.shape : 8, X (21932), 256
            aligned_emg = self.emg_aligner(
                emg_output,
                durations=None,                # optional: (B, Y) in mel frames // d_targets
                durations_in_mel=False,              # True if d_targets are mel-frame counts // True
                audio_sr=22050,
                hop_length=256,         # set this in model config (e.g., 256)
                emg_sr=1000,      # compute earlier (1000.XX)
                phoneme_enc=None,                 # provide phoneme queries for cross-attention (optional) // output
                target_len=None                     # optional fallback
            )
        output = self.phoneme_linear(output)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.mel_decoder_WoutStyle(output, mel_masks) #output: 16, 2366, 256 // style_vector: 16, 1, 256 --> output.shape: 16, 1000, 256
        output = self.mel_linear(output) #resulting output shape: 16, 1000, 80

        return (
            output,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
