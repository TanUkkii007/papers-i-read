# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

# commit hash
# 2d27126

from functools import wraps
import threading

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import tf_logging as logging

from hyperparams import Hyperparams as hp
from utils import *
from prepro import *

# Adapted from the `sugartensor` code.
# https://github.com/buriburisuri/sugartensor/blob/master/sugartensor/sg_queue.py
def producer_func(func):
    r"""Decorates a function `func` as producer_func.
    Args:
      func: A function to decorate.
    """
    @wraps(func)
    def wrapper(inputs, dtypes, capacity, num_threads):
        r"""
        Args:
            inputs: A inputs queue list to enqueue
            dtypes: Data types of each tensor
            capacity: Queue capacity. Default is 32.
            num_threads: Number of threads. Default is 1.
        """
        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))
        
        # enqueue function
        def enqueue_func(sess, op):
            # read data from source queue
            data = func(sess.run(inputs))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)
        
        # create FIFO queue
        queue = tf.FIFOQueue(capacity, dtypes=dtypes)

        # enqueue operation
        enqueue_op = queue.enqueue(placeholders)

        # create queue runner
        runner = _FuncQueueRunner(enqueue_func, queue, [enqueue_op] * num_threads)

        # return de-queue operation
        return queue.dequeue()
    
    return wrapper


class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                      queue_closed_exception_types, queue_runner_def)

    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    self.func(sess, enqueue_op) # call enqueue function
                except self._queue_closed_exception_types:
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1