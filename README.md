# UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook

In this work, we introduce **UniCodec**, a unified audio codec with a single codebook to support multi-domain audio data, including **speech**, **music**, and **sound**. 

![comparison](https://github.com/Jiang-Yidi/UniCodec/blob/main/comparison%20table.png)

To achieve this, we propose a **partitioned domain-adaptive codebook** method with **domain Mixture-of-Experts** strategy to capture the distinct characteristics of each audio domain. Furthermore, to enrich the semantic density of the codec **without auxiliary modules**, we propose a self-supervised mask prediction modeling approach. 

<div align=center>
<img src="https://github.com/Jiang-Yidi/UniCodec/blob/main/overview.png" width="50%">
</div>

As a single unified codec model, UniCodec achieves **superior subjective reconstruction performance** while maintaining a **high compression rate** in all three domains (speech/music/sound).

![main](https://github.com/Jiang-Yidi/UniCodec/blob/main/main%20result.png)
