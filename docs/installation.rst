Installation
============

There are two ways you can install ARLBench: PyPI and GitHub. The first one is the simplest way to install ARLBench, while the second one is the best way to install the latest version of ARLBench.
Either way, you will likely want to first create a virtual environment to install ARLBench into. This can be done by running the following command:

.. code-block:: bash

    conda create -n arlbench python=3.10
    conda activate arlbench

1. Using PyPI
-------------

This is the simplest way to install ARLBench. Just run the following command:

.. code-block:: bash

    pip install arlbench

2. Downloading from GitHub
--------------------------

This is the best way to install the latest version of ARLBench. First, clone the repository:

.. code-block:: bash

    git clone git@github.com:automl/arlbench.git
    cd arlbench

Then, install the package:

.. code-block:: bash

    make install

If you are on a Linux system and would like to use envpool environments, you can instead use:

.. code-block:: bash

    make install-envpool