# CHAR2WAV: END-TO-END SPEECH SYNTHESIS

Link: http://www.josesotelo.com/speechsynthesis/

Authors: Sotelo et al.

Institution: IIT Kanpur, INRS-EMT, CIFAR

Publication: Workshop track - ICLR 2017

Date: 2017

## Background Materials

- [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)

## What is this paper about?

- end-to-end model for speech synthesis named Char2Wav

## What is the motivation of this research?

Text-to-speech system consists of two stages and duty of the first stage called frontend is to  transforms linguistic features. Defining good linguistic features is often time-consuming and language specific. Char2Wav remove this process to integrate frontend and backend in end-to-end.

## What makes this paper different from previous research?

- Char2Wav can learn to produce audio directly from text.

## How this paper achieve it?

Char2Wav consists of two parts, reader and neural vocoder, corresponding to frontend and backend respectively.

### Reader

Attention based RNN generates sequence $Y = (y_1,...,y_T)$ conditioned on an input sequence X.
Y is a sequence of acoustic features and X is text or phoneme sequences.

The input X is preprocessed by an encoder, bidirectional RNN that outputs sequence $h = (h_1,...,hL)$

The output $y_i$ of $i$-th state of the generator is formulated as,

$y_i \sim \mathit{Generate(s_{i-1}, g_i)}$

where $g_i$ is weighted sum of $L$-length encoder output $h$.

$g_i = \sum_{j_1}^{L}\alpha_{i,j}h_j$

Here $\alpha_i \in G^L$ are attention weights formulated as,

$\alpha_i = \mathit{Attend(s_{i-1},\alpha_{i-1},h})$

The $i$-th state can be caluculated as,

$s_i = \mathit{RNN}(s_{i-1},g_i,y_i)$

Char2Wav uses a location-based attention mechanism developed by Graves (2013) for $\mathit{Attend}$ to calculate $\alpha$.


### Neural vocoder

Char2Wav uses SampleRNN (Mehri et al., 2016) for naural vocoder.

Unlike the SampleRNN research, Char2Wav uses conditional version of the model to map sequence of vocoder features to corresponding audio samples.

### Training

WORLD vocoder features are used as targets for the reader and as inputs for the neural vocoder.

## Dataset used in this study

Generated audio can be found at http://josesotelo.com/speechsynthesis/

## Implementations

- [sotelo/parrot](https://github.com/sotelo/parrot)

## Further Readings

- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
