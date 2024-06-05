Considerations for Seeding
============================

Seeding is important both on the level of RL algorithms as well as the AutoRL level. 
In general, we propose to use **three different set of random seeds** for training, validation, and testing.

For **training and validation**, ARLBench takes care of the seeding. When you pass a seed to the AutoRL Environment, it uses this seed for training but `seed + 1` for the validation during training.
We recommend to use seeds `0` - `9` for training and validation, i.e., by passing them to the AutoRL Environment for the tuning process.
You are of course free to increase this range, but we recommend to use **at least 10 different seeds** for reliable results.

When it comes to testing HPO methods, we provide a evaluation script in our examples. 
We propose to use seeds `100, 101, ...` here to make sure the method is tested on a different set of random seeds.
Here we suggest **three HPO runs as a minimum** even for stable optimizers - for consistent results with small confidence intervals, you should like aim for more runs.