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
from networks import encode, decode1, decode2
from modules import *
from data_load import get_batch
from utils import shift_by_one, visualize_attention, figure_to_tensor


class Graph:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.z, self.num_batch = get_batch()
            else:  # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, None))
                self.y = tf.placeholder(
                    tf.float32, shape=(None, None, hp.n_mels * hp.r))

            self.decoder_inputs = shift_by_one(self.y)
            with tf.variable_scope("net"):
                # Encoder
                self.memory = encode(
                    self.x, is_training=is_training)  # (N, T, E)

                # Decoder
                self.outputs1, attention_final_state = decode1(
                    self.decoder_inputs, self.memory,
                    is_training=is_training)  # (N, T', hp.n_mels*hp.r)
                
                alignment_history = attention_final_state.alignment_history.stack() # (decoder_timestep, batch_size, memory_size)
                self.alignment_history = tf.transpose(alignment_history, perm=[1,2,0]) # (batch_size, memory_size, decoder_timestep)

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
    char2idx, idx2char = load_vocab()
    g = Graph()
    print("Training Graph loaded")

    summary_g = tf.Graph()

    with g.graph.as_default():

        # Training
        sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0, summary_op=g.merged)

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
                    _, l1, l2, l, alignment_history, x = sess.run([g.train_op, g.mean_loss1, g.mean_loss2, g.mean_loss, g.alignment_history, g.x])
                    print("mean_loss1={}, mean_loss2={}, mean_loss={}".format(l1, l2, l))

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' %
                              (epoch, gs))
                
                with summary_g.as_default():
                    with tf.Session(graph=summary_g) as summary_sess:
                        visualize_attention(alignment_history[0], [idx2char[idx] + ' ' for idx in np.fromstring(x[0], np.int32)])
                        plot = figure_to_tensor()
                        attention_image = tf.summary.image("attention %d" % gs, plot)
                        merged = summary_sess.run(tf.summary.merge([attention_image]))
                        sv.summary_computed(sess, merged, gs)

if __name__ == '__main__':
    main()
    print("Done")