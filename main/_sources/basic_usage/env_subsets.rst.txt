The ARLBench Subsets
====================

We analyzed the hyperparameter landscapes of PPO, DQN and SAC on 20 environments to select a subset which allows for efficient benchmarking of AutoRL algorithms. 
This subset of 4-5 environments per algorithms matches the overall reward distribution across 128 hyperparameter configurations and 10 seeds::

.. image:: ../images/subsets.png
  :width: 800
  :alt: Environment subsets for PPO, DQN and SAC

In our experiments on GPU, all subset together should take about **1.5h to evaluate once**. 
This number will need to be multiplied by the number of RL seeds you want to evaluate on, the number of optimizer runs you consider as well as the optimization budget for the total runtime of your experiments.
If this full runtime is too long for your setup, you can also consider evaluating only a subset of algorithms - we strongly recommend you focus your benchmarking **on these exact environments**, however, to ensure you cover the space total landscape of RL behaviors well. 

The data generated for selecting these environments is available on `HuggingFace <https://huggingface.co/datasets/autorl-org/arlbench>`_ for you to use in your experiments.
For more information how the subset selection was done, please refer to our paper.
The examples in our GitHub repository show how you can evaluate your own method using these subsets. 