# UniCodec (ACL 2025 Main)

> [**UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook**](https://arxiv.org/abs/2502.20067)<br>
> [Yidi Jiang](https://scholar.google.com/citations?user=le6gC58AAAAJ&hl=en&oi=ao),[Qian Chen](https://scholar.google.com/citations?user=8eosmSQAAAAJ&hl=en&oi=ao),[Shengpeng Ji](https://scholar.google.com/citations?user=zuRaB-oAAAAJ&hl=en&oi=ao),[Yu Xi](https://scholar.google.com/citations?user=dszdUXYAAAAJ&hl=zh-CN),[Wen Wang](https://scholar.google.com.hk/citations?user=85Tj1OwAAAAJ&hl=en),[Chong Zhang](https://scholar.google.com.sg/citations?user=nqcBaoYAAAAJ&hl=en),[Xianghu Yue](https://scholar.google.com/citations?user=jWNJCDIAAAAJ&hl=en),[Shiliang Zhang](https://scholar.google.com/citations?user=BcWMSE4AAAAJ&hl=zh-CN),[Haizhou Li](https://colips.org/~eleliha/)<br>
> National University of Singapore; Tongyi Speech Lab<br>

In this work, we introduce **UniCodec**, a unified audio codec with a single codebook to support multi-domain audio data, including **speech**, **music**, and **sound**. 

![comparison](https://github.com/Jiang-Yidi/UniCodec/blob/main/comparison%20table.png)

To achieve this, we propose a **partitioned domain-adaptive codebook** method with **domain Mixture-of-Experts** strategy to capture the distinct characteristics of each audio domain. Furthermore, to enrich the semantic density of the codec **without auxiliary modules**, we propose a self-supervised mask prediction modeling approach. 

<div align=center>
<img src="https://github.com/Jiang-Yidi/UniCodec/blob/main/overview.png" width="50%">
</div>

As a single unified codec model, UniCodec achieves **superior subjective reconstruction performance** while maintaining a **high compression rate** in all three domains (speech/music/sound).

![main](https://github.com/Jiang-Yidi/UniCodec/blob/main/main%20result.png)

## Installation

```bash
pip3 install git+https://github.com/mesolitica/UniCodec-fix
```

## Encode decode

```python
from encodec.utils import convert_audio
from unicodec.decoder.pretrained import Unicodec
import torchaudio
import torch
config = 'configs/unicodec_frame75_10s_nq1_code16384_dim512_finetune.yaml'

# !wget https://huggingface.co/Yidiii/UniCodec_ckpt/resolve/main/unicode.ckpt
model = Unicodec.from_pretrained0802(config, 'unicode.ckpt')

wav, sr = torchaudio.load('husein-assistant-trim.mp3')
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])

# 0 for speech, 1 for music, 2 for audio based on https://github.com/mesolitica/UniCodec-fix/blob/main/encoder/quantization/simvq_moe.py#L161
_, discrete_code = model.encode_infer(wav, '2', bandwidth_id=bandwidth_id)
features = model.codes_to_features(discrete_code)

audio_out = model.decode(features, bandwidth_id=bandwidth_id)
```

Config and model also already mirrored at https://huggingface.co/huseinzol05/UniCodec-mirror

## Citation

```
@article{jiang2025unicodec,
  title={UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook},
  author={Jiang, Yidi and Chen, Qian and Ji, Shengpeng and Xi, Yu and Wang, Wen and Zhang, Chong and Yue, Xianghu and Zhang, ShiLiang and Li, Haizhou},
  journal={arXiv preprint arXiv:2502.20067},
  year={2025}
}
```
