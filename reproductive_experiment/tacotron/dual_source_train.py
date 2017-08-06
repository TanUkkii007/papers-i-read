# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

# commit hash
# 29a51a8

from __future__ import print_function
import tensorflow as tf
import numpy as np
import librosa
from tqdm import tqdm

import os

from hyperparams import Hyperparams as hp
from prepro import *
from networks import encode_vocab, dual_decode1, decode2
from modules import *
from data_load import get_dual_source_batch
from utils import shift_by_one
from prepro import load_vocab


class DualSourceAttentionGraph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                self.x1, self.x2, self.y, self.z, self.num_batch = get_dual_source_batch()
            else:  # Evaluation
                self.x1 = tf.placeholder(tf.int32, shape=(None, None))
                self.x2 = tf.placeholder(tf.int32, shape=(None, None))
                self.y = tf.placeholder(
                    tf.float32, shape=(None, None, hp.n_mels * hp.r))

            self.decoder_inputs = shift_by_one(self.y)
            with tf.variable_scope("net"):
                # Encoder 1
                # ToDo: use load_dual_source_vocabrary 
                char2idx_kana, idx2char_kana = load_vocab_ja_hiragana()
                self.memory1 = encode_vocab(
                    self.x1, char2idx_kana, idx2char_kana, is_training=is_training, scope="encoder1")  # (N, T, E)
                
                # Encoder 2
                # ToDo: use load_dual_source_vocabrary
                phone2idx, idx2phone = load_phone_ja()
                self.memory2 = encode_vocab(
                    self.x2, phone2idx, idx2phone, is_training=is_training, scope="encoder2")  # (N, T, E)

                # Decoder
                self.outputs1 = dual_decode1(
                    self.decoder_inputs, self.memory1, self.memory2,
                    is_training=is_training)  # (N, T', hp.n_mels*hp.r)
                self.outputs2 = decode2(
                    self.outputs1,
                    is_training=is_training)  # (N, T', (1+hp.n_fft//2)*hp.r)

            if is_training:
                # Loss
                if hp.loss_type == "l1":  # L1 loss
                    self.loss1 = tf.abs(self.outputs1 - self.y)
                    self.loss2 = tf.abs(self.outputs2 - self.z)
                else:  # L2 loss
                    self.loss1 = tf.squared_difference(self.outputs1, self.y)
                    self.loss2 = tf.squared_difference(self.outputs2, self.z)

                # Target masking
                if hp.target_zeros_masking:
                    self.loss1 *= tf.to_float(tf.not_equal(self.y, 0.))
                    self.loss2 *= tf.to_float(tf.not_equal(self.z, 0.))

                self.mean_loss1 = tf.reduce_mean(self.loss1)
                self.mean_loss2 = tf.reduce_mean(self.loss2)
                self.mean_loss = self.mean_loss1 + self.mean_loss2

                # Training Scheme
                self.global_step = tf.Variable(
                    0, name='global_step', trainable=False)
                learning_rate = tf.train.exponential_decay(
                    learning_rate=hp.lr,
                    global_step=self.global_step,
                    decay_steps=hp.decay_step,
                    decay_rate=hp.decay_rate,
                    staircase=False)
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(
                    self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss1', self.mean_loss1)
                tf.summary.scalar('mean_loss2', self.mean_loss2)
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()


def main():
    g = DualSourceAttentionGraph()
    print("Training Graph loaded")

    with g.graph.as_default():

        # Training
        sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0)

        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop(): break
                print("epoch={}".format(epoch))
                for step in tqdm(
                        range(g.num_batch),
                        total=g.num_batch,
                        ncols=70,
                        leave=False,
                        unit='b'):
                    sess.run(g.train_op)
                    l1, l2, l = sess.run([g.mean_loss1, g.mean_loss2, g.mean_loss])
                    print("mean_loss1={}, mean_loss2={}, mean_loss={}".format(l1, l2, l))

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' %
                              (epoch, gs))


if __name__ == '__main__':
    main()
    print("Done")