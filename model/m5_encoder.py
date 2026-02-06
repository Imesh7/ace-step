from transformer import T5ForConditionalGeneration, T5Tokenizer

class M5Encoder:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        

    def encode(self, x):
        inputs = self.tokenizer(x, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return outputs