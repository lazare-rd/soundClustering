import os
import librosa
import sys
import site

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class DataLoader():
    def __init__(self, data_dir):
        self.dir = data_dir
        self.mapping = {}
        self.load_data()
        
    def load_data(self):
        id = 0
        for root, dirs, files in os.walk(self.dir):
            for file in files:
                self.mapping[id] = os.path.join(root, file)
                id += 1

    def get_embedding(self):
        embeddings = []
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=16000)
            embedding = self.embed(y)
            embeddings.append(embedding)
        return

    def get_MFCC(self):
        mfccs = [] 
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfccs.append(mfcc)
        return mfccs

    def get_MelSpec(self):
        specs = [] 
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=None)
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            specs.append(spec)
        return specs


    def get_Prosod(self):
        pass

    def embed(self, y):
        model_name = "facebook/wav2vec2-base-960h"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)

        input_values = processor(y, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            embeddings = model(input_values).last_hidden_state

        return embeddings

    
dl = DataLoader('data/wav')
print(dl.mapping)
print(sys.executable)
print(librosa.__version__)
print(site.ENABLE_USER_SITE)
y, sr = librosa.load('data/wav/exemple.mp3', sr=None)
