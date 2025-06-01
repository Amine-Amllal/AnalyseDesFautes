Installation Guide
==================

This guide will help you set up VARS (Video Assistant Referee System) on your machine.

System Requirements
------------------

**Hardware Requirements:**

* **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3080 or higher)
* **RAM**: Minimum 16GB, recommended 32GB
* **Storage**: At least 100GB free space for datasets
* **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7 equivalent)

**Software Requirements:**

* **Operating System**: Windows 10/11, Ubuntu 18.04+, or macOS 10.15+
* **Python**: Version 3.9 or higher
* **CUDA**: Version 11.0 or higher (for GPU acceleration)

Environment Setup
-----------------

Using Conda (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Install Anaconda or Miniconda**

   Download from `Anaconda <https://www.anaconda.com/products/distribution>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

2. **Create Virtual Environment**

   .. code-block:: bash

      conda create -n vars python=3.9
      conda activate vars

3. **Install PyTorch with CUDA**

   Visit `PyTorch installation page <https://pytorch.org/get-started/locally/>`_ and select your configuration:

   .. code-block:: bash

      # Example for CUDA 11.8
      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

Using pip (Alternative)
~~~~~~~~~~~~~~~~~~~~~~

1. **Create Virtual Environment**

   .. code-block:: bash

      python -m venv vars_env
      
      # On Windows
      vars_env\Scripts\activate
      
      # On Linux/macOS
      source vars_env/bin/activate

2. **Install PyTorch**

   .. code-block:: bash

      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Core Dependencies
----------------

Install the required packages:

.. code-block:: bash

   # Install SoccerNet package
   pip install SoccerNet

   # Install other dependencies
   pip install -r requirements.txt

   # Install video processing library
   pip install pyav

**Main Dependencies:**

* ``torch`` >= 1.12.0
* ``torchvision`` >= 0.13.0
* ``SoccerNet`` >= 1.6.0
* ``PyQt5`` >= 5.15.0 (for interface)
* ``opencv-python`` >= 4.6.0
* ``numpy`` >= 1.21.0
* ``matplotlib`` >= 3.5.0

Interface Dependencies
--------------------

For the VARS interface application:

.. code-block:: bash

   pip install PyQt5
   pip install av

Download Model Weights
---------------------

Download the pre-trained model weights:

1. Access the model weights from `Google Drive <https://drive.google.com/drive/folders/1N0Lv-lcpW8w34_iySc7pnlQ6eFMSDvXn?usp=share_link>`_
2. Download ``14_model.pth.tar``
3. Place the file in the ``VARS interface/interface/`` directory

Dataset Access
--------------

To access the SoccerNet-MVFoul dataset:

1. **Fill NDA Form**

   Complete the Non-Disclosure Agreement at: `NDA Form <https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform>`_

2. **Download Dataset**

   .. code-block:: python

      from SoccerNet.Downloader import SoccerNetDownloader as SNdl
      
      mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
      mySNdl.downloadDataTask(
          task="mvfouls", 
          split=["train","valid","test","challenge"], 
          password="your_password_here"
      )

3. **High Resolution Option**

   For 720p videos, add ``version="720p"`` to the download arguments.

Verification
-----------

Test your installation:

.. code-block:: python

   import torch
   import torchvision
   import SoccerNet
   
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"SoccerNet version: {SoccerNet.__version__}")

Troubleshooting
--------------

**Common Issues:**

1. **CUDA not available**
   
   * Ensure NVIDIA drivers are up to date
   * Verify CUDA installation: ``nvcc --version``
   * Reinstall PyTorch with correct CUDA version

2. **Memory errors during training**
   
   * Reduce batch size in training scripts
   * Use gradient accumulation for effective larger batches
   * Enable mixed precision training

3. **Video codec issues**
   
   * Install additional codecs: ``conda install -c conda-forge ffmpeg``
   * On Windows, install K-Lite Codec Pack

4. **PyQt5 installation issues**
   
   * On Linux: ``sudo apt-get install python3-pyqt5``
   * On macOS: ``brew install pyqt5``

Docker Installation (Advanced)
-----------------------------

For containerized deployment:

.. code-block:: dockerfile

   FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   CMD ["python", "main.py"]

Build and run:

.. code-block:: bash

   docker build -t vars .
   docker run --gpus all -v /path/to/data:/data vars

Next Steps
---------

After successful installation:

1. Follow the :doc:`quickstart` guide
2. Explore the :doc:`dataset` documentation
3. Try :doc:`model_training` tutorials
4. Launch the :doc:`interface_usage` application
