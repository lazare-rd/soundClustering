import os
import librosa
import sys
import site

import torch
import numpy as np
#from transformers import Wav2Vec2Processor, Wav2Vec2Model

class DataLoader():
    def __init__(self, data_dir, target_sec):
        self.list_dir = data_dir
        self.mapping = {}
        self.ground_truth = {}
        self.load_data()
        self.TARGET_LEN = target_sec * 22050 if target_sec is not None else None
        self.troncate = True if target_sec is not None else False
        
    def load_data(self):
        id = 0
        j = 0
        for dir in self.list_dir:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    self.mapping[id] = os.path.join(root, file)
                    self.ground_truth[id] = j
                    id += 1
            j+=1

    def get_ground_truth(self):
        gt = []
        for i in range(len(self.ground_truth)):
            gt.append(self.ground_truth[i])
        return gt

    def get_MFCC(self):
        mfccs = [] 
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=22050)
            if self.TARGET_LEN is not None:
                if len(y) < self.TARGET_LEN:
                    y = np.pad(y, (0, self.TARGET_LEN - len(y)))
                else:
                    y = y[:self.TARGET_LEN]
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfccs.append(mfcc)
        return mfccs

    def get_MelSpec(self):
        specs = [] 
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=22050)
            if self.TARGET_LEN is not None:
                if len(y) < self.TARGET_LEN:
                    y = np.pad(y, (0, self.TARGET_LEN - len(y)))
                else:
                    y = y[:self.TARGET_LEN]
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            specs.append(spec)
        return specs


    def get_signal(self):
        signal = []
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=22050)
            if self.TARGET_LEN is not None:
                if len(y) < self.TARGET_LEN:
                    y = np.pad(y, (0, self.TARGET_LEN - len(y)))
                else:
                    y = y[:self.TARGET_LEN]
            signal.append(y)

        return signal

    def get_Prosod(self):
        pass

    """
    def get_embedding(self):
        embeddings = []
        for i in range(len(self.mapping)):
            y, sr = librosa.load(self.mapping[i], sr=16000)
            embedding = self.embed(y)
            embeddings.append(embedding)
        return embeddings
    
    def embed(self, y):
        model_name = "facebook/wav2vec2-base-960h"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)

        input_values = processor(y, return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            embeddings = model(input_values).last_hidden_state

        return embeddings
    """



if __name__ == "__main__":
    dl = DataLoader('data/mp4')
    print (dl.mapping)
    mfccs = dl.get_MelSpec()
    print (mfccs)
