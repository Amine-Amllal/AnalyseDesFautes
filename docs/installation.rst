Installation
============

To setup the environment for VARS, follow these steps:

1. Create and activate conda environment:

   .. code-block:: bash

      conda create -n vars python=3.9
      conda activate vars

2. Install PyTorch with CUDA (see https://pytorch.org/get-started/locally/).

3. Install dependencies:

   .. code-block:: bash

      pip install SoccerNet
      pip install -r requirements.txt
      pip install pyav
