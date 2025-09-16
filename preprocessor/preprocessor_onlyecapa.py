# import os
# import random
# import json
# import torch

# import tgt
# import librosa
# import numpy as np
# from scipy.interpolate import interp1d
# from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm
# import audio as Audio
# import torchaudio

# from speechbrain.pretrained import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
# random.seed(1234)


# class Preprocessor:
#     def __init__(self, config):
#         self.config = config
#         self.in_dir = config["path"]["raw_path"]
#         self.out_dir = config["path"]["preprocessed_path"]
#         self.val_size = config["preprocessing"]["val_size"]
#         self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
#         self.hop_length = config["preprocessing"]["stft"]["hop_length"]

#         self.STFT = Audio.stft.TacotronSTFT(
#             config["preprocessing"]["stft"]["filter_length"],
#             config["preprocessing"]["stft"]["hop_length"],
#             config["preprocessing"]["stft"]["win_length"],
#             config["preprocessing"]["mel"]["n_mel_channels"],
#             config["preprocessing"]["audio"]["sampling_rate"],
#             config["preprocessing"]["mel"]["mel_fmin"],
#             config["preprocessing"]["mel"]["mel_fmax"],
#         )

#     def build_from_path(self):
#         os.makedirs((os.path.join(self.out_dir, "xvect")), exist_ok=True)

#         print("Processing Data ...")
#         out = list()

#         # Compute pitch, energy, duration, and mel-spectrogram
#         speakers = {}
#         for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
#             speakers[speaker] = i
#             for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
#                 if ".wav" not in wav_name:
#                     continue

#                 basename = wav_name.split(".")[0]
#                 chapter = basename.split("_")[1]
#                 tg_path = os.path.join(
#                     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
#                 )
#                 if os.path.exists(tg_path):
#                     ret = self.process_utterance(speaker, chapter, basename)
#                     if ret is None:
#                         continue
#                     else:
#                         info = ret
#                     out.append(info)
#                 else:
#                     continue

#         return out

#     def process_utterance(self, speaker, chapter, basename):
#         wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
#         text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
#         tg_path = os.path.join(
#             self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
#         )

#         # Get alignments
#         textgrid = tgt.io.read_textgrid(tg_path)
#         phone, duration, start, end = self.get_alignment(
#             textgrid.get_tier_by_name("phones")
#         )
#         text = "{" + " ".join(phone) + "}"
#         if start >= end:
#             return None

#         # Read and trim wav files
#         wav, sr = librosa.load(wav_path)
#         wav = wav[
#             int(self.sampling_rate * start) : int(self.sampling_rate * end)
#         ].astype(np.float32)

#         # encode audio (i.e., wav) with pretrained xvect speaker verification
#         wav_org, sr = torchaudio.load(wav_path)
#         if sr != 16000:
#             signal = torchaudio.functional.resample(wav_org, sr, 16000)
#         xvect_embeddings = classifier.encode_batch(signal).squeeze(0) # shape: 1, 1, 192 --> 1, 192

#         # Read raw text
#         with open(text_path, "r") as f:
#             raw_text = f.readline().strip("\n")

#         xvect_filename = "{}-xvect-{}.npy".format(speaker, basename)
#         np.save(os.path.join(self.out_dir, "xvect", xvect_filename), xvect_embeddings)

#         return (
#             "|".join([basename, speaker, text, raw_text]),
#         )

#     def get_alignment(self, tier):
#         sil_phones = ["sil", "sp", "spn"]

#         phones = []
#         durations = []
#         start_time = 0
#         end_time = 0
#         end_idx = 0
#         for t in tier._objects:
#             s, e, p = t.start_time, t.end_time, t.text

#             # Trim leading silences
#             if phones == []:
#                 if p in sil_phones:
#                     continue
#                 else:
#                     start_time = s

#             if p not in sil_phones:
#                 # For ordinary phones
#                 phones.append(p)
#                 end_time = e
#                 end_idx = len(phones)
#             else:
#                 # For silent phones
#                 phones.append(p)

#             durations.append(
#                 int(
#                     np.round(e * self.sampling_rate / self.hop_length)
#                     - np.round(s * self.sampling_rate / self.hop_length)
#                 )
#             )

#         # Trim tailing silences
#         phones = phones[:end_idx]
#         durations = durations[:end_idx]

#         return phones, durations, start_time, end_time




import os
import random
import json
import torch

import tgt
import librosa
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import audio as Audio
import torchaudio
from model.yin import *

from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "ecapa")), exist_ok=True)

        print("Processing Data ...")
        out = list()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                chapter = basename.split("_")[1]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, chapter, basename)
                    if ret is None:
                        continue
                    else:
                        info = ret
                    out.append(info)
                else:
                    continue

        return out

    def process_utterance(self, speaker, chapter, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, sr = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # encode audio (i.e., wav) with pretrained ecapa speaker verification
        wav_org, sr = torchaudio.load(wav_path)
        if sr != 16000:
            signal = torchaudio.functional.resample(wav_org, sr, 16000)
        ecapa_embeddings = classifier.encode_batch(signal).squeeze(0) # shape: 1, 1, 192 --> 1, 192

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        ecapa_filename = "{}-ecapa-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "ecapa", ecapa_filename), ecapa_embeddings)

        return (
            "|".join([basename, speaker, text, raw_text]),
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time
