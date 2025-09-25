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

        self.mel_style_encoder = MelStyleEncoder(preprocess_config, model_config)
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

    def G(
            self,
            style_vector, # 28, 1, 128
            texts, # 28, 313
            src_masks, # 28, 313
            mel_masks, # 28, 2679
            max_mel_len,
            emg,
            p_targets=None, # 28, 314
            e_targets=None,
            d_targets=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
        ):
        output = self.phoneme_encoder_WoutStyle(texts, src_masks) # 16, 171, 256 --> phoneme_encoder_WoutStyle로 바꿨을 때의 shape: 16, X, 256
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
        output = self.phoneme_linear(output) # 16, 257, 256

        (
            output, # shape: 
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
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
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
        emg,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        output = self.phoneme_encoder_WoutStyle(texts, src_masks) # 16, 171, 256 --> phoneme_encoder_WoutStyle로 바꿨을 때의 shape: 16, X, 256
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

        output, mel_masks = self.mel_decoder_WoutStyle(output, mel_masks) #output: 16, 2366, 256 // style_vector: 16, 1, 256 --> output.shape: 16, 1000, 256
        output = self.mel_linear(output) #resulting output shape: 16, 1000, 80

        return (
            output,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    def meta_learner_1(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        raw_quary_texts=None,
        quary_texts=None,
        quary_src_lens=None,
        max_quary_src_len=None,
        quary_d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        quary_mel_lens = quary_d_targets.sum(dim=-1)
        max_quary_mel_len = max(quary_mel_lens).item()
        quary_src_masks = get_mask_from_lengths(quary_src_lens, max_quary_src_len)
        quary_mel_masks = get_mask_from_lengths(quary_mel_lens, max_quary_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            _,
            _,
            _,
            d_rounded_adv,
            mel_lens_adv,
            mel_masks_adv,
        ) = self.G(
            style_vector,
            quary_texts,
            quary_src_masks,
            quary_mel_masks,
            max_quary_mel_len,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )

        D_s = self.D_s(self.style_prototype, speakers, output, mel_masks_adv)

        quary_texts = self.phoneme_encoder.src_word_emb(quary_texts)
        D_t = self.D_t(self.variance_adaptor.upsample, quary_texts, output, max(mel_lens_adv).item(), mel_masks_adv, d_rounded_adv)

        (
            G,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.G(
            style_vector,
            texts,
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

        return (
            D_s,
            D_t,
            G,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    def meta_learner_2(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        raw_quary_texts=None,
        quary_texts=None,
        quary_src_lens=None,
        max_quary_src_len=None,
        quary_d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        quary_mel_lens = quary_d_targets.sum(dim=-1)
        max_quary_mel_len = max(quary_mel_lens).item()
        quary_src_masks = get_mask_from_lengths(quary_src_lens, max_quary_src_len)
        quary_mel_masks = get_mask_from_lengths(quary_mel_lens, max_quary_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (
            output,
            _,
            _,
            _,
            d_rounded_adv,
            mel_lens_adv,
            mel_masks_adv,
        ) = self.G(
            style_vector,
            quary_texts,
            quary_src_masks,
            quary_mel_masks,
            max_quary_mel_len,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )

        texts = self.phoneme_encoder.src_word_emb(texts)
        D_t_s = self.D_t(self.variance_adaptor.upsample, texts, mels, max_mel_len, mel_masks, d_targets)

        quary_texts = self.phoneme_encoder.src_word_emb(quary_texts)
        D_t_q = self.D_t(self.variance_adaptor.upsample, quary_texts, output, max(mel_lens_adv).item(), mel_masks_adv, d_rounded_adv)

        D_s_s = self.D_s(self.style_prototype, speakers, mels, mel_masks)
        D_s_q = self.D_s(self.style_prototype, speakers, output, mel_masks_adv)

        # Get Style Logit
        w = style_vector.squeeze() # [B, H]
        style_logit = torch.matmul(w, self.style_prototype.weight.contiguous().transpose(0, 1)) # [B, K]

        return D_t_s, D_t_q, D_s_s, D_s_q, style_logit
