Objectives in ARLBench
======================

ARLBench allows to configure the objectives you'd like to use for your AutoRL methods. 
These are selected as a list of keywords in the configuration of the AutoRL Environment, e.g. like this:

.. code-block:: bash

    python arlbench.py autorl.objectives=["reward_mean"]

The following objectives are available at the moment:
- reward_mean: the mean evaluation reward across a number of evaluation episodes
- reward_std: the standard deviation of the evaluation rewards across a number of evaluation episodes
- runtime: the runtime of the training process
- emissions: the CO2 emissions of the training process