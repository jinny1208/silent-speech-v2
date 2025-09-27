import argparse
import os
from functools import partial
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import MetaStyleSpeechLossMain
from dataset import Dataset
import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None, loss_len=5, emgFlag=False):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "Usethis-val2UnandSeeSpk-sameButErasedSeenandUnseenLabel.txt", preprocess_config, train_config, sort=False, drop_last=False
    ) # Usethis-val2UnandSeeSpk-sameButErasedSeenandUnseenLabel.txt or V5-val_merged_filelist-noDup-noMisalignedSpeakerID.txt
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(dataset.collate_fn, emgFlag=train_config["emgInput"]["emgFlag"]),
    )

    # Get loss function
    Loss = MetaStyleSpeechLossMain(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(loss_len)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                # pdb.set_trace()
                if emgFlag:
                    output = model(*(batch[2:-5]))
                else:
                    # inject None in place of emg
                    no_emg_batch = list(batch[2:-5])
                    no_emg_batch.insert(7, None)   # 6th position = where emg would be
                    output = model(*no_emg_batch)

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means[:5]])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message
