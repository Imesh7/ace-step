# ACE-Step
This is ACE-Step paper implementation (Unofficial) from scratch. This paper's goal is to build a model to generate text-to-song & features.

[Paper](https://arxiv.org/pdf/2506.00045)


### Key features

- Linear Attention
- DiT (Diffussion transformer) implemented
- Converted songs into mel-spectrogram
- `Tags` -> mT5 encoder
- `Lyrics` -> VoiceBPE Tokenizer
- Cross Attention
- RoPE implement
- Implement Training pipeline


### Environment Setup

1. Clone the repository:
```
git@github.com:Imesh7/ace-step.git
```

2. Setup enviornment

``` 
conda env create --name ace-step -f environment.yml
```

3. Activate the conda environment:
```
conda activate ace-step
```



#### Dependency

Install Dependecies 
```
conda install --file requirements.txt
```


### Folder structure

```
├─── model
│     ├─── autoencoder
|     |        ├─── autoencoder.py
|     |        ├─── encoder.py
|     |        └─── encoder.py
|     |
│     ├─── DiT
|     |      └─── dit.py
|     |
│     └─── transformer
|     |        ├─── attention.py
|     |        ├─── cross_attention.py
|     |        └─── mix_feed_forward.py
|     |
|     ├─── RoPE.py
|     └─── m5_encoder.py
|     
├─── notebook
├─── tests
├─── train.py
└─── inference.py
```

### What's next
- Train the Autoencoder & Upload it to Huggineface
- Implement Encoders for music
- Flash Attention 3


### Overall Architecture

<img width="1000" height="350" alt="image" src="https://github.com/user-attachments/assets/d7617052-c02f-41ee-b304-81392ffd0e28" />
