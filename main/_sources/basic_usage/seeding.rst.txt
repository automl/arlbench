Considerations for Seeding
============================

Seeding is important both on the level of RL algorithms as well as the AutoRL level. In general, we propose to use three different random seeds for training, validation, and testing.
For training and validation, ARLBench takes care of the seeding. When you pass a seed to the AutoRL Environment, it uses this seed for training but `seed + 1` for the validation during training.
We recommend to use seeds `0` - `9` for training and validation, i.e., by passing them to the AutoRL Environment for the tuning process.

When it comes to testing HPO methods, we provide a evaluation script in our examples. We propose to use seeds `100, 101, ...` here to make sure the method is tested on a different set of random seeds.