class AceStepPipeline:
    def __init__(self, autoencoder, dit, optimizer, max_timestamps=1000):
        self.autoencoder = autoencoder
        self.dit = dit
        self.optimizer = optimizer
        self.max_timesteps = max_timestamps
        self.device = next(autoencoder.parameters()).device
 
    def encode(self, input):
        return self.autoencoder.encode(input)