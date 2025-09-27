import os
import json

import torch
import numpy as np

import hifigan
from model import StyleSpeech, ScheduledOptimMain, ScheduledOptimDisc


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = StyleSpeech(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        # --- filter checkpoint dict manually ---
        ckpt_state = ckpt["model"]
        model_state = model.state_dict()

        # keep only overlapping + same shape keys
        filtered_state = {}
        for k, v in ckpt_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f"[Skip] {k} from checkpoint (not in model or shape mismatch)")

        # load safely
        msg = model.load_state_dict(filtered_state, strict=False)

        if msg.missing_keys:
            print(f"[Warning] Missing keys when loading: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"[Warning] Unexpected keys in checkpoint: {msg.unexpected_keys}")
        
        ## original checkpoint loading
        # model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim_main = ScheduledOptimMain(
            model, train_config, model_config, args.restore_step
        )
        scheduled_optim_disc = ScheduledOptimDisc(
            model, train_config
        )
        if args.restore_step:
            scheduled_optim_main.load_state_dict(ckpt["optimizer_main"])
            scheduled_optim_disc.load_state_dict(ckpt["optimizer_disc"])
        model.train()
        return model, scheduled_optim_main, scheduled_optim_disc

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
