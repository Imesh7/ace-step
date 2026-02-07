# ACE-Step
This is ACE-Step paper implementation (Unofficial) from scratch. This paper's goal is to build a model to generate text-to-song & features.

[Paper](https://arxiv.org/pdf/2506.00045)


### Key features

- Multi-Head Attention
- DiT(Diffussion transformer) implemented
- Converted songs into mel-spectrogram
- `Tags` -> mT5 encoder
- `Lyrics` -> VoiceBPE Tokenizer

### What's next
- Cross Attention
- Train the Autoencoder & Upload it to huggineface
- Implement Encoders for music
- Flash Attention 3


### Overall Architecture

<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/d7617052-c02f-41ee-b304-81392ffd0e28" />
