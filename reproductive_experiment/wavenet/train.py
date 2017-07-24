'''
The MIT License (MIT) Copyright (c) 2016 Igor Babuschkin

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
"""Training script for the WaveNet network on the VCTK corpus.
This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
import tensorflow as tf
from tensorflow.python.client import timeline

from model import WaveNetModel

BATCH_SIZE = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='How many wav files to process at once. Default: ' +
        str(BATCH_SIZE) + '.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=DATA_DIRECTORY,
        help='The directory containing the VCTK corpus.')
    parser.add_argument(
        '--store_metadata',
        type=bool,
        default=METADATA,
        help='Whether to store advanced debugging information '
        '(execution time, memory consumption) for use with '
        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument(
        '--logdir',
        type=str,
        default=None,
        help='Directory in which to store the logging '
        'information for TensorBoard. '
        'If the model already exists, it will restore '
        'the state and will continue training. '
        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument(
        '--logdir_root',
        type=str,
        default=None,
        help='Root directory to place the logging '
        'output and generated model. These are stored '
        'under the dated subdirectory of --logdir_root. '
        'Cannot use with --logdir.')
    parser.add_argument(
        '--restore_from',
        type=str,
        default=None,
        help='Directory in which to restore the model from. '
        'This creates the new model under the dated directory '
        'in --logdir_root. '
        'Cannot use with --logdir.')
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=CHECKPOINT_EVERY,
        help='How many steps to save each checkpoint after. Default: ' +
        str(CHECKPOINT_EVERY) + '.')
    parser.add_argument(
        '--num_steps',
        type=int,
        default=NUM_STEPS,
        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate for training. Default: ' + str(LEARNING_RATE) +
        '.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters. Default: ' +
        WAVENET_PARAMS + '.')
    parser.add_argument(
        '--sample_size',
        type=int,
        default=SAMPLE_SIZE,
        help='Concatenate and cut audio samples to this many '
        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument(
        '--l2_regularization_strength',
        type=float,
        default=L2_REGULARIZATION_STRENGTH,
        help='Coefficient in the L2 regularization. '
        'Default: False')
    parser.add_argument(
        '--silence_threshold',
        type=float,
        default=SILENCE_THRESHOLD,
        help='Volume threshold below which to trim the start '
        'and the end from the training set samples. Default: ' +
        str(SILENCE_THRESHOLD) + '.')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=optimizer_factory.keys(),
        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=MOMENTUM,
        help='Specify the momentum to be '
        'used by sgd or rmsprop optimizer. Ignored by the '
        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument(
        '--histograms',
        type=_str_to_bool,
        default=False,
        help='Whether to store histogram summaries. Default: False')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help=
        'Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument(
        '--max_checkpoints',
        type=int,
        default=MAX_TO_KEEP,
        help='Maximum amount of checkpoints that will be kept alive. Default: '
        + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def get_default_logdir(logdir_root):
    logdir = ps.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from VCTK corpus.