ARLBench and Different AutoRL Paradigms
=======================================

In this chapter, we elaborate on the relationship between ARLBench in various AutoRL Paradigms.

Hyperparameter Optimization (HPO)
---------------------------------
(Static) Hyperparameter optimization is one of the core use cases of ARLBench. As stated in our examples, ARLBench supports all kinds of black-box optimizers to perform hyperparameter optimization for RL.

Dynamic Algorithm Configuration (DAC)
-------------------------------------
When it comes to dynamic approaches, ARLBench supports different kinds of optimization techniques that adapt the current hyperparameter configuration during training. As stated in the examples,
this can be done using the CLI or the AutoRL Environment. Using checkpointing, trainings can be continued seamlessly which allows for flexible dynamic approaches.

Neural Architecture Search (NAS)
--------------------------------
In addition to HPO, ARLBench supports NAS approaches that set the size of hidden layers and activation functions. However, as of now this is limited to these two architecture hyperparameters.
In the future, ARLBench could be extended by more powerful search space interfaces for NAS.

Meta-Gradients
--------------
As of now, ARLBench does not include meta-gradient based approaches for AutoRL. However, we allow for reactive dynamic approaches that use the gradient informatio during training to select the next hyperparameter configuration as stated in our examples.