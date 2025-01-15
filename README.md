# NeuralComp

In neuroscience research, a theory known as [multiplexing](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00103-7) posits that when presented with multiple stimuli together, individual neurons can switch over time between encoding each member of the stimulus ensemble, causing a fluctuating pattern of firing rates. `NeuralComp` is an R package for testing and analyzing rate fluctuations of multiplexing neurons under a Bayesian paradigm. 

**Corresponding Paper**
  - [Modeling Neural Switching via Drift-Diffusion Models](https://arxiv.org/abs/2410.00781)

## Data format

To gain insight into multiplexing, scientists often collect _triplets_ of data consisting of: spike trains recorded under an $A$ stimulus, spike trains recorded under a $B$ stimulus, and spike trains recorded under both the $A$ and $B$ stimuli (we will refer to this as the $AB$ condition). An example of a triplet from [Caruso et al.](https://www.nature.com/articles/s41467-018-05121-8) can be seen below. Given a triplet, we wish to infer whether the neuron utilizes multiplexing to encode both stimuli and, if so, at what time-scale does the neuron switch between encoding the two stimuli.

![Example of a triplet](./images/Caruso_Task_Annotated.pdf)

_Practical Consideration: We often necessitate at least 5 trials from each condition in the triplet._

## Models
In the corresponding paper, we propose a mechanistic statistical model for multiplexing. The model uses the integrate-and-fire framework as the basis, which means that we assume that the spikes occur as a result of a latent membrane voltage process hitting a threshold. In this work, the latent membrane voltage process is assumed to be a drift-diffusion process. By assuming a perfect-integrator integrate-and-fire model, we have that the hitting times of the drift-diffusion process are inverse Gaussian distributed. The relationship between the latent drift-diffusion process and the spikes can be seen below. We will model the $A$ and $B$ condition spike trains using these types of inverse Gaussian point processes, and will posit that multiplexing occurs due to competition between the $A$ condition drift-diffusion process and the $B$ drift-diffusion process. To control the overall firing rate and rate of switching, we will assume that there is some inhibition (or penalty for switching) in the form of a time delay ($\delta$) on one of the processes. The effect of $\delta$ can be seen in the figures below. This type of model results in a poential neural motif that could lead to this type of behavior (as seen in subfigure B).

![**Subfigure A:** Visualization of how the latent drift-diffusion processes relate to the observed spike trains. **Subfigure B:** Potential neural motif that leads to this type of firing behavior](./images/Drift_diffusion_point_process.pdf)

**Alternative Model:** We propose a alternative model that characterizes alternative encoding schemes (normalization, winner-take-all, ect.) with some level of generality. This model assumes that the $AB$ condition spike trains can be modeled using another inverse Gaussian point process with parameters not necessarily relating to the $A$ process or the $B$ process.

## Conceptual Idea
Given our multiplexing model and alternative model, we use WAIC to determine whether the data supported the occurance of multiplexing in the $AB$ trials. The general schematic of the analysis can be seen in the figure below.

![Conceptual diagram of the proposed analysis. This figure also illustrates how the spike train analysis relates to spike count approaches, which are often used in these settings.](./images/Conceptual_Idea.pdf)



## Associated Repositories
  1. [Simulation Studies](https://github.com/ndmarco/NeuralComp_Sim_Study)
  2. [Case Study](https://github.com/ndmarco/NeuralComp_Case_Study)

## Related Multiplexing Papers
#### Statistical Modeling
  1. [Spike Count Analysis for MultiPlexing Inference (SCAMPI)](https://www.biorxiv.org/content/10.1101/2024.09.14.613077v1)
  2. [Analyzing second order stochasticity of neural spiking under stimuli-bundle exposure](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373042)
#### Neuroscience Papers
  1. [Signal switching may enhance processing power of the brain](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00103-7)
  2. [Single neurons may encode simultaneous stimuli by switching between activity patterns](https://www.nature.com/articles/s41467-018-05121-8)
  3. [Coordinated multiplexing of information about separate objects in visual cortex](https://elifesciences.org/articles/76452)
  4. [Multiple objects evoke fluctuating responses in several regions of the visual pathway](https://elifesciences.org/articles/91129)
  5. [Sensitivity and specificity of a Bayesian single trial analysis for time varying neural signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC8425354)
