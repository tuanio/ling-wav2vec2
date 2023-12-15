import json
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import phoneme_tokenizer, encode_phone
from .augmentation import SpeedPerturbation, AdaptiveSpecAugment
from transformers import AutoTokenizer

SAMPLING_RATE = 16000


class SupervisedDataset(Dataset):
    def __init__(self, data_path, do_augment=False):
        super().__init__()
        self.data = json.load(open(data_path, encoding="utf-8"))
        self.n_fft = 512
        self.hop_len = 128
        self.n_mels = 80
        self.cal_mel = MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            n_mels=self.n_mels,
        )
        self.do_augment = do_augment
        self.spec_aug = AdaptiveSpecAugment()
        self.speed_pertub = SpeedPerturbation(SAMPLING_RATE)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.data[idx]["path"])

        if self.do_augment:
            wav = self.speed_pertub(wav)

        mel = self.cal_mel(wav).permute(0, 2, 1)  # (1, seq len, 80)
        mel_size = mel.size(1)

        if self.do_augment:
            mel, _ = self.spec_aug(mel, torch.LongTensor([mel_size]))

        mel = mel[0]

        energy = mel.sum(dim=1).unsqueeze(1)

        # concat mel with energy
        # (sequence length, 81)
        mel = torch.cat([mel, energy], dim=1)

        # get pich only, remove NCCF
        # (sequence length, 1)
        pitch = torchaudio.functional.compute_kaldi_pitch(
            wav,
            sr,
            frame_length=self.n_fft / SAMPLING_RATE * 1000,
            frame_shift=self.hop_len / SAMPLING_RATE * 1000,
            snip_edges=False,
        )[0, :, :1]

        if mel.size(0) != pitch.size(0):
            # odd length
            pitch = torch.cat([pitch, pitch[-1:, :]], dim=0)

        phoneme = phoneme_tokenizer(self.data[idx]["transcript"], sep=" ")
        ids = encode_phone(phoneme)

        tonal = torch.LongTensor(self.data[idx]["tonal"])

        return mel, mel_size, ids, len(ids), pitch, tonal, len(tonal)


def collate_fn(batch):
    mels = pad_sequence([i[0] for i in batch], batch_first=True)
    mels_len = torch.LongTensor([i[1] for i in batch])
    ids = pad_sequence([i[2] for i in batch], batch_first=True)
    ids_len = torch.LongTensor([i[3] for i in batch])
    pitches = pad_sequence([i[4] for i in batch], batch_first=True)
    tonals = pad_sequence([i[5] for i in batch], batch_first=True)
    tonal_len = torch.LongTensor([i[6] for i in batch])
    return mels, mels_len, ids, ids_len, pitches, tonals, tonal_len
