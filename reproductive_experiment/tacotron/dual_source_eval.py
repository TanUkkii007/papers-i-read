# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import codecs
import copy
import os

import librosa
from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
import numpy as np
from prepro import *
import tensorflow as tf
from dual_source_train import DualSourceAttentionGraph
from utils import *


def eval():
    # Load graph
    g = DualSourceAttentionGraph(is_training=False)
    print("Graph loaded")

    # Load data
    X1, X2 = load_dual_source_eval_data()  # texts
    # ToDo: use load_dual_source_vocabrary
    char2idx_kana, idx2char_kana = load_vocab_ja_hiragana()
    # ToDo: use load_dual_source_vocabrary
    phone2idx, idx2phone = load_phone_ja()

    ah1 = g.attention_final_state.state1_alignment_history
    alignment_history1 = ah1.gather(tf.range(0, ah1.size()))
    ah2 = g.attention_final_state.state2_alignment_history
    alignment_history2 = ah2.gather(tf.range(0, ah2.size()))

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            # Get model
            mname = open(hp.logdir + '/checkpoint',
                         'r').read().split('"')[1]  # model name

            timesteps = 100  # Adjust this number as you want
            outputs1 = np.zeros((hp.num_samples, timesteps, hp.n_mels * hp.r),
                                np.float32)
            for j in range(timesteps):
                _outputs1 = sess.run(g.outputs1,
                                     {g.x1: X1,
                                      g.x2: X2,
                                      g.y: outputs1})
                outputs1[:, j, :] = _outputs1[:, j, :]
            outputs2 = sess.run(g.outputs2, {g.outputs1: outputs1})

            alignment_history1 = sess.run(alignment_history1,
                                          {g.x1: X1,
                                           g.x2: X2,
                                           g.y: outputs1})
            alignment_history2 = sess.run(alignment_history2,
                                          {g.x1: X1,
                                           g.x2: X2,
                                           g.y: outputs1})

    # Generate wav files
    if not os.path.exists(hp.outputdir): os.mkdir(hp.outputdir)
    with codecs.open(hp.outputdir + '/text.txt', 'w', 'utf-8') as fout:
        for i, (x1, x2, s, a1, a2) in enumerate(
                zip(X1, X2, outputs2, alignment_history1, alignment_history2)):
            # write text
            fout.write(
                str(i) + "\t" + "".join(idx2char_kana[idx]
                                        for idx in np.fromstring(x1, np.int32))
                + "\n")

            s = restore_shape(s, hp.win_length // hp.hop_length, hp.r)

            # generate wav files
            if hp.use_log_magnitude:
                audio = spectrogram2wav(np.power(np.e, s)**hp.power)
            else:
                s = np.where(s < 0, 0, s)
                audio = spectrogram2wav(s**hp.power)
            write(hp.outputdir + "/{}_{}.wav".format(mname, i), hp.sr, audio)

            visualize_attention(a1, [
                idx2char_kana[idx] for idx in np.fromstring(x1, np.int32)
            ])
            save_figure(hp.outputdir + "/{}_{}_attention1.png".format(
                mname, i))
            visualize_attention(
                a2, [idx2phone[idx] for idx in np.fromstring(x2, np.int32)])
            save_figure(hp.outputdir + "/{}_{}_attention2.png".format(
                mname, i))


if __name__ == '__main__':
    eval()
    print("Done")