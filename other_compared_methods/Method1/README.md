# Compared Method 1

This is an implementation of the model in paper "Bio-inspired digit recognition using reward-modulated spike-timing-dependent plasticity in deep convolutional networks" ([Link to paper](https://doi.org/10.1016/j.patcog.2019.05.015)).

This implementation is based on code from [SpykeTorch/MozafariDeep.py at master · miladmozafari/SpykeTorch · GitHub](https://github.com/miladmozafari/SpykeTorch/blob/master/MozafariDeep.py)

Some changes are listed below:

* Network sizes changed.

* Threshold values tuned.

* Input processing changed.

* Two tests, i.e., input noise and network parameter noise, added.

## Note

To run the code `train.py`, you need to get the `SpykeTorch` simulator from the original repo [GitHub - miladmozafari/SpykeTorch: High-speed simulator of convolutional spiking neural networks with at most one spike per neuron.](https://github.com/miladmozafari/SpykeTorch) and put it in folder `SpykeTorch`.
