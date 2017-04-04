# FAST WAVENET GENERATION ALGORITHM

Link: https://arxiv.org/abs/1611.09482

Authors: Paine et al. 2016

Institution: University of Illinois, IBM Thomas J. Watson Research Center


## Background Materials

- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

## What is this paper about?

Algorithm refinement of WaveNet.

## What makes this paper different from previous research?

- Computation complexity is reduced from O(L^2) to O(L)

## How this paper achieve it?

A single output sample is generated from cached state of previous timesteps.
The cached state can be viewed as "recurrent" state with analogy of RNN.

## Dataset used in this study


## Implementations

- https://github.com/tomlepaine/fast-wavenet
- https://github.com/ibab/tensorflow-wavenet ("naive implementation" this paper refers to)

## Further Readings
