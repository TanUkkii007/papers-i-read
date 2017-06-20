# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

# commit hash
# 95ed868

from __future__ import print_function
import tensorflow as tf
import numpy as np
import librosa

import copy
from hyperparams import Hyperparams as hp

def get_spectrograms(sound_file):
    '''Extracts melspectrogram and log magnitude from given `sound_file`.
    Args:
      sound_file: A string. Full path of a sound file.
    Returns:
      Transposed S: A 2d array. A transposed melspectrogram with shape of (T, n_mels)
      Transposed magnitude: A 2d array.Has shape of (T, 1+hp.n_fft//2)
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=None) # or set sr to hp.sr.

    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
    
    # magnitude spectrogram
    magnitude = np.abs(D) #(1+n_fft/2, T)

    # power spectrogram
    power = magnitude**2 #(1+n_fft/2, 1)

    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels) #(n_mels, T)

    return np.transpose(S.astype(np.float32)), np.transpose(magnitude.astype(np.float32)) # (T, n_mels), (T, 1+n_fft/2)

def reduce_frames(array, step, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:
      arry: A 2d array with shape of [T, C]
      step: An int. Overlapping span.
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    T = array.shape[0]
    num_padding = (step*r) - (T % (step*r)) if T % (step*r) !=0 else 0

    array = np.pad(array, [[0, num_padding], [0,0]], 'constant', constant_values=(0, 0))
    T, C = array.shape
    sliced = np.split(array, list(range(step, T, step)), axis=0)

    started = False
    for i in range(0, len(sliced), r):
        if not started:
            reshaped = np.hstack(sliced[i:i+r])
            started = True
        else:
            reshaped = np.vstack((reshaped, np.hstack(sliced[i:i+r])))
    
    return reshaped