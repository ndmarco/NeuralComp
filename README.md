# NeuralComp

In neuroscience research, a theory known as [multiplexing](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00103-7) posits that when presented with multiple stimuli together, individual neurons can switch over time between encoding each member of the stimulus ensemble, causing a fluctuating pattern of firing rates. `NeuralComp` is an R package for testing and analyzing rate fluctuations of multiplexing neurons under a Bayesian paradigm. 

**Corresponding Paper**
  - [Modeling Neural Switching via Drift-Diffusion Models](https://arxiv.org/abs/2410.00781)

## Data format

To gain insight into multiplexing, scientists often collect _triplets_ of data consisting of: spike trains recorded under an $A$ stimulus, spike trains recorded under a $B$ stimulus, and spike trains recorded under both the $A$ and $B$ stimuli (we will refer to this as the $AB$ condition). An example of a triplet can be seen below. Given a triplet, we wish to infer whether the neuron utilizes multiplexing to encode both stimuli and, if so, at what time-scale does the neuron switch between encoding the two stimuli.
