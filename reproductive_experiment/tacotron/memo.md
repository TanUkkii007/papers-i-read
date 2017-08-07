## ATR 503

### 2017-07-29
num_epochs = 40
sanity_check=True
lr = 0.0005
decay_step = 500

- a low learning rate is preferable
- loss2 is relatively high than loss2
- slow training rate. ~2.3batch/s.

### 2017-08-05
reverse_input = True
num_epochs = 40
sanity_check=True
lr = 0.0005
decay_step = 500
decay_rate=0.96

- suspended at epoch 29 because I found a bug
- was able to generate higher quality audio than the previous experiment though

### 2017-08-07
source: kana
reverse_input = True
num_epochs = 1000
sanity_check=False
lr = 0.0005
decay_step = 500
decay_rate=0.96
batch/ per epoch = 14

- final loss: mean_loss1=0.86, mean_loss2=0.45, mean_loss=1.3
- failed to generate distinct sounds
- failed to learn alignments

## SIWIS

### 2017-07-30
num_epochs = 40
sanity_check=True
lr = 0.0010
decay_step = 300

- a high learning rate is preferable
- loss2 is relatively high than loss2

### 2017-07-31
5 days training
num_epochs = 1000
sanity_check=False
lr = 0.0001
decay_rate=1.00
batch/ per epoch = 139

- loss 1 stayed high and decreased slowly with high variance
- failed to learn when to stop speaking


## bible

### 2017-07-31
num_epochs = 40
sanity_check=False
lr = 0.0001
decay_step = 385
batch/ per epoch = 385

- failed to train

## atr503_dual

### 2017-08-05
source: kana-phone
reverse_input = True
num_epochs = 1000
sanity_check=False
lr = 0.0005
decay_step = 500
decay_rate=0.96
batch/ per epoch = 15

- slow training rate. ~3.2 batch/s.
- final loss: mean_loss1=0.93, mean_loss2=0.56, mean_loss=1.48
- failed to generate distinct sounds, but the quality is better than single source one