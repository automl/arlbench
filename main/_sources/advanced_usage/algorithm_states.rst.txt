Using the ARLBench States
==========================

In addition to providing different objectives, ARLBench also provides insights into the target algorithms' internal states. This is done using so called `StateFeatures`.
As of now, we implement the `GradInfo` state feature which returns the norm the gradients observed during training.

The used state features can be defined using the `state_features` key in the config passed to the AutoRL Environment. Please include `grad_info` in this list if you want to use this state feature for your approach.