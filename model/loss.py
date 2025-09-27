import torch
import torch.nn as nn


class MetaStyleSpeechLossMain(nn.Module):
    """ Meta-StyleSpeech Loss for naive StyleSpeech and Step 1 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(MetaStyleSpeechLossMain, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.alpha = train_config["optimizer"]["alpha"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        if len(inputs[6:]) == 12:
            (
                mel_targets,
                _,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
                _,
                _,
                _,
                _,
                _,
            ) = inputs[6:]
        else:
            (
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
                _,
                _,
                _,
                _,
                _,
            ) = inputs[6:]
        (
            mel_predictions,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        mel_targets.requires_grad = False

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)

        alpha = 1
        recon_loss = alpha * (mel_loss)
        total_loss = (
            recon_loss
        )

        return (
            total_loss,
            mel_loss,
        )

