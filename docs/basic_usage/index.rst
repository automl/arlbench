Benchmarking AutoRL Methods
============================

ARLBench provides an basis for benchmarking different AutoRL methods. This section of the documentation focuses on the prominent aspect of black-box hyperparameter optimization, since it's the simplest usecase of ARLBench.
We discuss the structure of ARLBenchmark, the currently supported objectives, the environment subsets and search spaces we provide and the seeding of the experiments in their own subpages. 

The most important question, however, is how to actually use ARLBench in your experiments. This is the workflow we propose which you can also see in our examples:

1. Decide **which RL algorithms** you choose as your HPO targets. In the best case, you will use all three: PPO, DQN and SAC. You should also decide on the number of runs per algorithm you can afford to run (we recommend at least 10).
2. Decide **which AutoRL methods** you want to benchmark. Also set a number of runs per AutoRL method (we recommend 3 at the very least, ideally more).
3. Decide **which objectives** you want to optimize for. We provide a variety of objectives you can select one or more from.
4. **Use the pre-defined search spaces** in your setup. If there is a good reason to deviate from these search spaces, please report this alongside your results.
5. **Execute your experiments** for all combinations your defined - use this same setup for any baselines you compare against.
5. **Evaluate** the best found configuration on the environment test seeds and report this result.


In-depth Information on:
^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   objectives
   env_subsets
   seeding
   options