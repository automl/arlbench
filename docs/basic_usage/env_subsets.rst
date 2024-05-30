The ARLBench Subsets
====================

We analyzed the hyperparameter landscapes of PPO, DQN and SAC on 20 environments to select a subset which allows for efficient benchmarking of AutoRL algorithms. These are the resulting subsets:

.. image:: ../images/subsets.png
  :width: 800
  :alt: Environment subsets for PPO, DQN and SAC

We strongly recommend you focus your benchmarking on these exact environments to ensure you cover the space total landscape of RL behaviors well. 
The data generated for selecting these environments is available on `HuggingFace <https://huggingface.co/datasets/autorl-org/arlbench>`_ for you to use in your experiments.
For more information how the subset selection was done, please refer to our paper.