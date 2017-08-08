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
import matplotlib.pyplot as plt

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
    y, sr = librosa.load(sound_file, sr=None)  # or set sr to hp.sr.

    # stft. D: (1+n_fft//2, T)
    D = librosa.stft(
        y=y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length)

    # magnitude spectrogram
    magnitude = np.abs(D)  #(1+n_fft/2, T)

    # power spectrogram
    power = magnitude**2  #(1+n_fft/2, 1)

    # mel spectrogram
    S = librosa.feature.melspectrogram(S=power, n_mels=hp.n_mels)  #(n_mels, T)

    return np.transpose(S.astype(np.float32)), np.transpose(
        magnitude.astype(np.float32))  # (T, n_mels), (T, 1+n_fft/2)

def shift_by_one(inputs):
    '''Shifts the content of `inputs` to the right by one 
      so that it becomes the decoder inputs.
      
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
    
    Returns:
      A 3d tensor with the same shape and dtype as `inputs`.
    '''
    return tf.concat((tf.zeros_like(inputs[:, :1, :]), inputs[:, :-1, :]), 1)

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
    num_padding = (step * r) - (T % (step * r)) if T % (step * r) != 0 else 0

    array = np.pad(
        array, [[0, num_padding], [0, 0]], 'constant', constant_values=(0, 0))
    T, C = array.shape
    sliced = np.split(array, list(range(step, T, step)), axis=0)

    started = False
    for i in range(0, len(sliced), r):
        if not started:
            reshaped = np.hstack(sliced[i:i + r])
            started = True
        else:
            reshaped = np.vstack((reshaped, np.hstack(sliced[i:i + r])))

    return reshaped


def spectrogram2wav(spectrogram):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    spectrogram = spectrogram.T # [f, t]
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length) # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est)) # [f, t]
        X_best = spectrogram * phase # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window='hann')

def restore_shape(array, step, r):
    '''Reduces and adjust the shape and content of `arry` according to r.
    
    Args:
      arry: A 2d array with shape of [T, C]
      step: An int. Overlapping span.
      r: Reduction factor
     
    Returns:
      A 2d array with shape of [-1, C*r]
    '''
    T, C = array.shape
    sliced = np.split(array, list(range(step, T, step)), axis=0)

    started = False
    for s in sliced:
        if not started:
            restored = np.vstack(np.split(s, r, axis=1))
            started = True
        else:
            restored = np.vstack((restored, np.vstack(np.split(s, r, axis=1))))
    
    # Trim zero paddings
    restored = restored[:np.count_nonzero(restored.sum(axis=1))]
    return restored


def visualize_attention(alignment_history, memory_label):
    fig, ax = plt.subplots()
    pcm = ax.pcolor(alignment_history)
    fig.colorbar(pcm, ax=ax)
    plt.yticks(np.arange(0.5, len(memory_label), 1.0), memory_label)


def save_figure(file_name):
    plt.savefig(file_name)