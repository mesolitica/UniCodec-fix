# --coding:utf-8--
import os,typing
from torch import nn
from typing import List
from audiotools import AudioSignal
from audiotools import STFTParams
from encodec.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import Unicodec
import time
import logging
from decoder.modules import safe_log


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
            self,
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)

            x_mag = x.magnitude
            y_mag = y.magnitude
            if x_mag.shape[-1] != y_mag.shape[-1]:
                length = min(x_mag.shape[-1],y_mag.shape[-1])
                x_mag = x_mag[:,:,:,:length]
                y_mag = y_mag[:,:,:,:length]

            loss += self.log_weight * self.loss_fn(
                x_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mag, y_mag)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(
            self,
            n_mels: List[int] = [150, 80],
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            mel_fmin: List[float] = [0.0, 0.0],
            mel_fmax: List[float] = [None, None],
            window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
                self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self, sample_rate: int = 24000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True, power=1,
        )
    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))
        # breakpoint()
        if mel_hat.shape[-1] != mel.shape[-1]:
            length = min(mel_hat.shape[-1],mel.shape[-1])
            mel_hat = mel_hat[:,:,:length]
            mel = mel[:,:,:length]

        loss = torch.nn.functional.l1_loss(mel, mel_hat)
        return loss

stft_loss = MultiScaleSTFTLoss()
mel_loss = MelSpecReconstructionLoss()

device1=torch.device('cuda:0')

input_path = "./data/infer/large_data_domain"
out_folder = './result/test_unified'
ll="libritts"


tmptmp=out_folder+"/"+ll

os.system("rm -r %s"%(tmptmp))
os.system("mkdir -p %s"%(tmptmp))

# 自己数据模型加载
config_path = "./configs/***.yaml"
model_path = "./checkpoints/***.ckpt"

codec = Unicodec.from_pretrained0802(config_path, model_path)
codec = codec.to(device1)

with open(input_path,'r') as fin:
    x=fin.readlines()


features_all=[]


##############Generate audio output##################
count = 0
sum_stft_loss = 0
sum_mel_loss = 0
for i in range(len(x)):
    wav_path, domain = x[i].split()
    if domain == '2':
        count += 1
        wav, sr = torchaudio.load(wav_path)
        wav = convert_audio(wav, sr, 24000, 1) 
        # if sr != 24000:
        #     wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)

        bandwidth_id = torch.tensor([0]).to(device1) 
        wav=wav.to(device1)
        print(count)

        features,discrete_code= codec.encode_infer(wav, domain, bandwidth_id=bandwidth_id)
        # features_all.append(features)

        audio_out = codec.decode(features, bandwidth_id=bandwidth_id)   

        ################MEL LOSS & STFT LOSS
        melloss = mel_loss(wav.cpu(),audio_out.cpu())
        stftloss = stft_loss(AudioSignal(wav,24000),AudioSignal(audio_out, 24000))


        sum_stft_loss += stftloss
        sum_mel_loss += melloss

print("*************STFT LOSS",sum_stft_loss/count)
print("*************MEL LOSS",sum_mel_loss/count)







