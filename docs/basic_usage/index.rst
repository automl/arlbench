Benchmarking AutoRL Methods
============================

.. toctree::
   :hidden:
   :maxdepth: 2
   objectives
   env_subsets
   seeding



ARLBench provides an basis for benchmarking different AutoRL methods. This section of the documentation focuses on the prominent aspect of black-box hyperparameter optimization, since it's the simplest usecase of ARLBench.
We discuss the structure of ARLBenchmark, the currently supported objectives, the environment subsets and search spaces we provide and the seeding of the experiments in their own subpages. 
The most important question, however, is how to actually use ARLBench in your experiments. This is the workflow we propose:

1. Decide which RL algorithms you choose as your HPO targets. In the best case, you will use all three: PPO, DQN and SAC.
2. Decide which AutoRL methods you want to benchmark. 
3. Decide which objectives you want to optimize for. We provide a variety of objectives you can select one or more from.
4. Use the pre-defined search spaces to run your AutoRL method for several runs. If there is a good reason to deviate from these search spaces, please report this alongside your results.
5. Evaluate the best found configuration on the environment test seeds and report this result.

