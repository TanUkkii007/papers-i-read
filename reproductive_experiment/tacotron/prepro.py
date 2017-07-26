#/usr/bin/python3
# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

# commit hash
# 9782c18

import numpy as np

from hyperparams import Hyperparams as hp
import re
import os
import csv
import codecs


def load_vocab():
    if hp.data_set == 'bible':
        return load_vocab_en()
    elif hp.data_set == 'atr503':
        return load_vocab_ja_hiragana()
    else:
        raise ValueError('unknown data set')


def load_vocab_en():
    vocab = "EG abcdefghijklmnopqrstuvwxyz'"  # E: Empty. ignore G
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


def load_vocab_ja_hiragana():
    vocab = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉっゃゅょー、。"
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def create_train_data():
    if hp.data_set == 'bible':
        return create_train_data_bible()
    elif hp.data_set == 'atr503':
        return create_train_data_atr503()
    else:
        raise ValueError('unknown data set')

def create_train_data_bible():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.bible_text_file, 'rb', 'utf-8'))
    for row in reader:
        sound_fname, text, duration = row
        sound_file = hp.bible_sound_fpath + "/" + sound_fname + ".wav"
        text = re.sub(r"[^ a-z]", "", text.strip().lower())

        if hp.min_len <= len(text) <= hp.max_len:
            texts.append(
                np.array([char2idx[char]
                          for char in text], np.int32).tostring())
            sound_files.append(sound_file)

    return texts, sound_files


def create_train_data_atr503():
    # Load vocabulary
    char2idx, idx2char = load_vocab_ja_hiragana()

    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.atr503_text_file, 'rb', 'utf-8'))
    for row in reader:
        sound_index, text_mixed, text_hiragana = row
        sound_fname = "nitech_jp_atr503_m001_" + sound_index
        sound_file = hp.atr503_sound_fpath + "/" + sound_fname + ".wav"

        if hp.min_len <= len(text_hiragana) <= hp.max_len:
            texts.append(
                np.array([char2idx[char]
                          for char in text_hiragana], np.int32).tostring())
            sound_files.append(sound_file)

    return texts, sound_files


def load_train_data():
    """We train on the whole data but the last num_samples."""
    texts, sound_files = create_train_data()
    if hp.sanity_check: # We use a single mini-batch for training to overfit it.
        texts, sound_files = texts[:hp.batch_size]*1000, sound_files[:hp.batch_size]*1000
    else:
        texts, sound_files = texts[:-hp.num_samples], sound_files[:-hp.num_samples]
    return texts, sound_files


def load_eval_data():
    """We evaluate on the last num_samples."""
    texts, _ = create_train_data()
    if hp.sanity_check: # We generate samples for the same texts as the ones we've used for training.
        texts = texts[:hp.batch_size]
    else:
        texts = texts[-hp.num_samples:]

    X = np.zeros(shape=[len(texts), hp.max_len], dtype=np.int32)
    for i, text in enumerate(texts):
        _text = np.fromstring(text, np.int32)  # byte to int
        X[i, :len(_text)] = _text

    return X