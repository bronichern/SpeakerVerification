import random

from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
from feature_exctractor import get_file_mel, get_two_files_mel, get_stft, sample_files

class CelebLoader(Dataset):
    def __init__(self, path, train):#, feat_func, *args, **kwargs):
        with open(path,"rb") as f:
            self.alike, self.diff = pickle.load(f)
        # if train:
        #     print("concatenating val")
        #     with open("valset","rb") as f:
        #         alike2, diff2 = pickle.load(f)
        #     self.alike = pd.concat([self.alike,alike2])
        #     self.diff = pd.concat([self.diff, diff2])
        # self.data, self.labels = self.generate_set()
        self.batch = 32
        self.train = train
        # self.data, self.labels, self.name_2_label = load_set_features(set_path, feat_func, args, kwargs)

    def __getitem__(self, index):
        file1, file2 = None, None
        if not self.train:
            if index < len(self.alike):
                pair = self.alike.iloc[index]
                label = 1
            else:
                pair = self.diff.iloc[index%len(self.diff)]
                label = 0
            file1, file2 = pair.file1, pair.file2
        else:
            if np.random.randint(0,2) == 1:
                pair = self.alike.sample(1)
                label = 1
            else:
                pair = self.diff.sample(1)
                label = 0
            file1, file2 = pair.file1.values[0], pair.file2.values[0]
        wav1, wav2 = self.load_tauple_files(file1,file2)

        return wav1, wav2, label

    def load_tauple_files(self, file1, file2):
        path_prefix = "/data/ronic/"
        # These could be replaced by a feature extraction function such as mfcc,stft,melspectrogram
        # file1_wav, file2_wav, sr = sample_files(path_prefix + file1, path_prefix +file2)
        # file1_wav = get_stft(file1_wav, sr[0])
        # file2_wav = get_stft(file2_wav, sr[1])
        # file1_wav = get_file_mel(path_prefix+file1)#sf.read(file1)
        # file2_wav = get_file_mel(path_prefix+file2) #sf.read(file2)
        file1_wav, file2_wav = get_two_files_mel(path_prefix + file1, path_prefix +file2)
        # This is data augmentation. It's an alternative to padding - or could be intertwined with
        # a feature extraction function.
        # Between the two files: sample from file with the max. len , contiguous min.len frames.
        # if len(file1_wav) < len(file2_wav):
        #     start_idx = np.random.randint(0, len(file2_wav) - len(file1_wav))
        #     file2_wav = file2_wav[start_idx:start_idx + len(file1_wav)]
        # elif len(file2_wav) < len(file1_wav):
        #     start_idx = np.random.randint(0, len(file1_wav) - len(file2_wav))
        #     file1_wav = file1_wav[start_idx:start_idx + len(file2_wav)]
        return file1_wav, file2_wav

    def __len__(self):
        return len(self.alike.file1.values)+len(self.diff.file1.values)

# cl = CelebLoader("voxceleb/trainset")
# for i in range(10):
#     cl.__getitem__(0)