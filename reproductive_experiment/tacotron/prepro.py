#/usr/bin/python3
# -*- coding: utf-8 -*-

'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

# commit hash
# 6009936

import numpy as np
import librosa

from hyperparams import Hyperparams as hp
import glob
import re
import os
import csv
import codecs

def load_vocab():
    vocab = "E abcdefghijklmnopqrstuvwxyz'" # E: Empty
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def create_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.text_file, 'rb', 'utf-8'))
    for row in reader:
        sound_fname, text, duration = row
        sound_file = hp.sound_fpath + "/" + sound_fname + ".wav"
        text = re.sub(r"[^ a-z]", "", text.strip().lower())

        if len(text) <= hp.max_len:
            texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
            sound_files.append(sound_file)
    
    return texts, sound_files