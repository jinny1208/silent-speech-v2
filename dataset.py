import json
import math
import os
import random
import pdb
import glob
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D, expand
from torchvision.io import read_video
from torchvision.transforms.functional import rgb_to_grayscale


random.seed(1234)


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text, self.speaker_to_ids = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx] # '0_5-5_86_audio_clean'
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        query_idx = random.choice(self.speaker_to_ids[speaker]) # Sample the query text
        raw_quary_text = self.raw_text[query_idx]
        query_phone = np.array(text_to_sequence(self.text[query_idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        quary_duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(self.speaker[query_idx], self.basename[query_idx]),
        )
        quary_duration = np.load(quary_duration_path)
        emg_path = os.path.join(
            self.preprocessed_path,
            "emg",
            "{}.npy".format(basename.replace('_audio_clean', '_emg')),
        )
        emg = np.load(emg_path)
        # video_path_candidates = glob.glob(os.path.join(
        #     self.preprocessed_path, 
        #     "talkinghead",
        #     f"*{basename}*_VC_audio_roi.mp4")
        # )
        # if len(video_path_candidates)>1:
        #     video_path_candidates = random.choice(video_path_candidates)
        # if video_path_candidates:
        #     video, audio, info = read_video(video_path_candidates, pts_unit="sec")  # (T, H, W, C)
            # video = video.float() / 255.0
            # video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

            # # Check and crop if needed
            # if video.shape[2:] == (512, 512):
            #     video = self.center_crop(video, 96, 96)  # (T, 3, 96, 96)
            # # Convert to grayscale
            # roi_features = rgb_to_grayscale(video)  # (T, 1, 96, 96)
            # roi_features = roi_features.detach().cpu().numpy()
        
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "quary_text": query_phone,
            "raw_quary_text": raw_quary_text,
            "mel": mel,            
            "emg": emg,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "quary_duration": quary_duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            speaker_to_ids = dict()
            for i, line in enumerate(f.readlines()):
                n, s, t, r, TalkHeadspeakerID = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if s not in speaker_to_ids:
                    speaker_to_ids[s] = [i]
                else:
                    speaker_to_ids[s] += [i]
            return name, speaker, text, raw_text, speaker_to_ids
        
    def center_crop(video, crop_h, crop_w):
        """Crop center region of shape (T, C, H, W) to (T, C, crop_h, crop_w)."""
        _, _, h, w = video.shape
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return video[:, :, top:top+crop_h, left:left+crop_w]

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        quary_texts = [data[idx]["quary_text"] for idx in idxs]
        raw_quary_texts = [data[idx]["raw_quary_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        quary_durations = [data[idx]["quary_duration"] for idx in idxs]
        emg = [data[idx]["emg"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        quary_text_lens = np.array([text.shape[0] for text in quary_texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        quary_texts = pad_1D(quary_texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        quary_durations = pad_1D(quary_durations)
        emg = pad_2D(emg)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            emg,
            pitches,
            energies,
            durations,
            raw_quary_texts,
            quary_texts,
            quary_text_lens,
            max(quary_text_lens),
            quary_durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class BatchInferenceDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        emg_path = os.path.join(
            self.preprocessed_path,
            "emg",
            "{}.npy".format(basename.replace('_audio_clean', '_emg')),
        )
        emg = np.load(emg_path)
        # pdb.set_trace()
        return (basename, speaker_id, phone, raw_text, mel, emg, pitch, energy, duration)
 
    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r, TalkHeadspeakerID = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        mels = [d[4] for d in data]
        emg = [d[5] for d in data]
        pitches = [d[6] for d in data]
        energies = [d[7] for d in data]
        durations = [d[8] for d in data]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        ref_infos = list()
        for _, (m, p, e, d) in enumerate(zip(mels, pitches, energies, durations)):
            if self.pitch_feature_level == "phoneme_level":
                pitch = expand(p, d)
            else:
                pitch = p
            if self.energy_feature_level == "phoneme_level":
                energy = expand(e, d)
            else:
                energy = e
            ref_infos.append((m.T, pitch, energy))

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        emg = pad_2D(emg)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            emg,
            ref_infos,
        )
