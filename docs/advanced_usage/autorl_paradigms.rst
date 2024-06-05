ARLBench and Different AutoRL Paradigms
=======================================

Since there are various AutoRL paradigms in the literature, we mention how ARLBench relates to each one.

Hyperparameter Optimization (HPO)
---------------------------------
Hyperparameter optimization is one of the core use cases of ARLBench. As stated in our examples, ARLBench supports all kinds of black-box optimizers to perform hyperparameter optimization for RL.
This can also be done in a dynamic fashion in ARLBench.

Dynamic Algorithm Configuration (DAC)
-------------------------------------
When it comes to dynamic approaches, ARLBench supports different kinds of optimization techniques that adapt the current hyperparameter configuration during training. As stated in the examples,
this can be done using the CLI or the AutoRL Environment. In DAC specifically, however, the hyperparameter controller learns to adapt hyperparameters based in an algorithm state. 
This is supported in ARLBench, but not implemented extensively just yet. At the moment, we only offer a limited amount of gradient features, which might not be enough to learn a reliable hyperparameter controller.
Since DAC has not been applied to RL in this manner yet, however, we are not yet sure which other features are necessary to make DAC work in the context of RL.

Neural Architecture Search (NAS)
--------------------------------
In addition to HPO, ARLBench supports NAS approaches that set the size of hidden layers and activation functions. However, as of now this is limited to these two architecture hyperparameters.
Most NAS approaches actually focus on more elaborate search spaces to find architectures tailored to a usecase. This line of research is not very prominent in the context of RL yet, unfortunately.
We hope ARLBench can support such research in the future by extending to standard NAS search spaces like DARTS or novel RL-specific ones.

Meta-Gradients
--------------
As of now, ARLBench does not include meta-gradient/second order optimization based approaches for AutoRL. 
However, we allow for reactive dynamic approaches that use the gradient informatio during training to select the next hyperparameter configuration as stated in our examples.
Through this interface, we jope to be able to provide an option for second order gradient computation in the future.