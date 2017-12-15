# Acronym Disambiguation using Word2GM (Word to Gaussian Mixture)

We approach the problem of acronym disambiguation in a subset of the MSH dataset (todo: reference) employing the Word2GM implementation of *[Athiwaratkun and Wilson](https://arxiv.org/abs/1704.08424), Multimodal Word Distributions, ACL 2017*.

This repository is a fork of the [original implementation](https://github.com/benathi/word2gm) with the following modifications:
- the source code was made compatible to TensorFlow 1.4
- added code for preprocessing the MSH dataset and performing acronym disambiguation

## Licenses
To achieve TensorFlow 1.4 compatabilty, the SkipGram custom-op kernel from a [TensorFlow tutorial repository](https://github.com/tensorflow/models/tree/master/tutorials/embedding) as referenced the [Tensorflow word2vec tutorial](https://www.tensorflow.org/tutorials/word2vec) was copied, which is suplied under the Apache Licence v2 (find the full license text in *APACHE_LICENSE*). 

The original code and all modifications are provided under the terms of the BSD 3-Clause License (see file *License*).

## Dependencies
Python 3, Tensorflow 1.4

[ggplot](https://github.com/yhat/ggplot.git)
```
pip install -U ggplot
# or 
conda install -c conda-forge ggplot
# or
pip install git+https://github.com/yhat/ggplot.git
```

## Usage Guide
Below are the steps for training and visualization.

1.1. Obtain the dataset

1.2. Preprocess (assuming that the MSH dataset resides in *MSH_location* in *.arff* format:
``` python  preprocess_MSH_corpus.py MSH_location```

1.3. Compile the custom-op and train:
```
./compile_word2vec_ops.sh

python word2gm_trainer.py --num_mixtures 2 --train_data data/msh_train.txt --spherical --embedding_size 50 --epochs_to_train 200 --var_scale 0.05 --save_path modelfiles/msh-k2-lr05-v05-e200-ss3-adg --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 3 --batch_size 2048 --max_to_keep 10 --checkpoint_interval 500 --window_size 10
# or simply calling ./train_msh.sh
```
See at the end of page for details on training options.

1.4. Note that the model will be saved at modelfiles/msh-k2-lr05-v05-e200-ss3-adg. The code to analyze the model and visualize the results is in **Analyze MSH Model.ipynb**. See model API below.


1.5. The ```Word2GM``` class in file **word2gm_loader.py** contains method ```visualize_embedding()``` which prepares the word embeddings to be visualized by TensorFlow's Tensorboard. It is invoked during execution of the **Analyze MSH Model.ipynb** notebook mentioned above.

Once the embeddings are prepared, the visualization can be done by shell command:
```
tensorboard --logdir=modelfiles/msh-k2-lr05-v05-e200-ss3-adg_emb --port=6006
```
Then, navigate the browser to (http://localhost/6006) (or a url of the appropriate machine that has the model) and click at the **Embeddings** tab. Note that the **logdir** folder is the "**original-folder**" + "_emb".

1.6. The script **evaluate.py** performs acronym disambiguation using the trained model and the test set split off the MSH corpus by **preprocess_MSH_corpus.py**. You have to pass in the locations of both:
```
python evaluate.py modelfiles/msh-k2-lr05-v05-e200-ss3-adg data/msh_test_with_labels.txt
```
The script uses the **find_best_cluster** method in **word2gm_loader.py** from the original implementation of word2gm to determine the mixture component of an acronym most likely for a given context. Since this method disregards the fact that words are modelled as distribution mixtures to some extent by performing only pair-wise comparisons of modes for all mixture pairs between acronym and context word, we provide an alternative implementation in the **disambiguate_posterior** branch of this repository.

## Visualization
The Tensorboard embeddings visualization tools (please use Firefox or Chrome) allow for nearest neighbors query, in addition to PCA and t-sne visualization. We use the following notation: *x:i* refers to the *i*th mixture component of word 'x'. For instance, querying for 'bank:0' yields 'river:1', 'confluence:0', 'waterway:1' as the nearest neighbors, which means that this component of 'bank' corresponds to river bank. On the other hand, querying for 'bank:1' gives the nearest neighbors 'banking:1', 'banker:0', 'ATM:0', which indicates that this component of 'bank' corresponds to financial bank.


## Training Options

```
arguments:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Directory to write the model and training summaries.
                        (required)
  --train_data TRAIN_DATA
                        Training text file. (required)
  --embedding_size EMBEDDING_SIZE
                        The embedding dimension size.
  --epochs_to_train EPOCHS_TO_TRAIN
                        Number of epochs to train. Each epoch processes the
                        training data once completely.
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --batch_size BATCH_SIZE
                        Number of training examples processed per step (size
                        of a minibatch).
  --concurrent_steps CONCURRENT_STEPS
                        The number of concurrent training steps.
  --window_size WINDOW_SIZE
                        The number of words to predict to the left and right
                        of the target word.
  --min_count MIN_COUNT
                        The minimum number of word occurrences for it to be
                        included in the vocabulary.
  --subsample SUBSAMPLE
                        Subsample threshold for word occurrence. Words that
                        appear with higher frequency will be randomly down-
                        sampled. Set to 0 to disable.
  --statistics_interval STATISTICS_INTERVAL
                        Print statistics every n seconds.
  --summary_interval SUMMARY_INTERVAL
                        Save training summary to file every n seconds (rounded
                        up to statistics interval).
  --checkpoint_interval CHECKPOINT_INTERVAL
                        Checkpoint the model (i.e. save the parameters) every
                        n seconds (rounded up to statistics interval).
  --num_mixtures NUM_MIXTURES
                        Number of mixture component for Mixture of Gaussians
  --spherical [SPHERICAL]
                        Whether the model should be spherical of diagonalThe
                        default is spherical
  --nospherical
  --var_scale VAR_SCALE
                        Variance scale
  --ckpt_all [CKPT_ALL]
                        Keep all checkpoints(Warning: This requires a large
                        amount of disk space).
  --nockpt_all
  --norm_cap NORM_CAP   The upper bound of norm of mean vector
  --lower_sig LOWER_SIG
                        The lower bound for sigma element-wise
  --upper_sig UPPER_SIG
                        The upper bound for sigma element-wise
  --mu_scale MU_SCALE   The average norm will be around mu_scale
  --objective_threshold OBJECTIVE_THRESHOLD
                        The threshold for the objective
  --adagrad [ADAGRAD]   Use Adagrad optimizer instead
  --noadagrad
  --loss_epsilon LOSS_EPSILON
                        epsilon parameter for loss function
  --constant_lr [CONSTANT_LR]
                        Use constant learning rate
  --noconstant_lr
  --wout [WOUT]         Whether we would use a separate wout
  --nowout
  --max_pe [MAX_PE]     Using maximum of partial energy instead of the sum
  --nomax_pe
  --max_to_keep MAX_TO_KEEP
                        The maximum number of checkpoint files to keep
  --normclip [NORMCLIP]
                        Whether to perform norm clipping (very slow)
  --nonormclip

```
