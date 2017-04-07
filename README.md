# TensorFlow Part-of-Speech Tagger

Simple part-of-speech tagger implemented using a feedforward network in
Tensorflow.

Done as a homework project for the Natural Language Understanding course at ETH
Zurich, taught by Prof. Thomas Hofmann and Dr. Massimiliano Ciaramita. Code
is based on a skeleton provided with the homework by Florian Schmidt.


## Usage

First, train the model using `train.py`:

```
$ python train.py
```

Once you're happy with how well trained the model is, run `demo.py`, input a
sentence, and see the model annotate it with
[Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
part-of-speech tags:

```
$ python demo.py
Enter a sentence to be annotated:
The Carterfone is a device invented by Thomas Carter

Loading saved vocabulary...
Generating tensors...
Your sentence, annotated:
The/DT Carterfone/NNP is/VBZ a/DT device/NN invented/VBN by/IN Thomas/NNP Carter/NNP
```
