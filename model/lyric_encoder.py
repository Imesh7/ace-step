import torch.nn as nn
from transformer import T5ForConditionalGeneration, T5Tokenizer

# In here I just used tramsformer tokeizer it hould be `Voice BPE` tokenizer
# In here voice means it just a normal tokenizer not related to any voice model
# trained with some custom text data

# I just used T5 tokenizer and model for simplicity
# to get the real implementation please refer to `Voice BPE` tokenizer

# # Install Coqui TTS
# pip install coqui-tts

# # OR for development
# git clone https://github.com/idiap/coqui-ai-TTS
# cd coqui-ai-TTS
# pip install -e 

# from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# # Initialize with vocab file (from XTTS model)
# tokenizer = VoiceBpeTokenizer(vocab_file="path/to/vocab.json")

class LyricEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        return outputs
