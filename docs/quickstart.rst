Quick Start Guide
================

This guide will get you up and running with VARS in just a few minutes.

Prerequisites
------------

Before starting, ensure you have:

* Python 3.9+ installed
* CUDA-compatible GPU (recommended)
* At least 16GB RAM
* Completed the :doc:`installation` steps

Basic Workflow
-------------

The VARS system follows this workflow:

1. **Data Preparation**: Download and organize the SoccerNet-MVFoul dataset
2. **Model Training**: Train the multi-view neural network (optional - pre-trained weights available)
3. **Evaluation**: Test the model performance on validation/test sets
4. **Interface Usage**: Use the interactive GUI for real-time analysis

1. Dataset Download
------------------

First, download the SoccerNet-MVFoul dataset:

.. code-block:: python

   from SoccerNet.Downloader import SoccerNetDownloader as SNdl
   
   # Initialize downloader
   mySNdl = SNdl(LocalDirectory="path/to/your/dataset")
   
   # Download dataset (requires NDA password)
   mySNdl.downloadDataTask(
       task="mvfouls", 
       split=["train", "valid", "test", "challenge"], 
       password="your_nda_password"
   )

.. note::
   You need to fill out the NDA form to get access credentials. See :doc:`installation` for details.

2. Quick Model Training
----------------------

Train a model with default settings:

.. code-block:: bash

   cd "VARS model"
   python main.py --path "path/to/dataset"

**Key Parameters:**

* ``--path``: Path to your dataset directory
* ``--batch_size``: Training batch size (default: 8)
* ``--epochs``: Number of training epochs (default: 15)
* ``--pre_model``: Backbone architecture (default: "mvit_v2_s")

**Example with Custom Settings:**

.. code-block:: bash

   python main.py \
       --path "C:/dataset/mvfouls" \
       --batch_size 4 \
       --epochs 20 \
       --pooling_type "attention" \
       --pre_model "mvit_v2_s"

3. Model Inference
-----------------

Run inference on test data:

.. code-block:: bash

   python main.py \
       --pooling_type "attention" \
       --start_frame 63 \
       --end_frame 87 \
       --fps 17 \
       --path "path/to/dataset" \
       --pre_model "mvit_v2_s" \
       --path_to_model_weights "14_model.pth.tar"

**Frame Selection Tips:**

* Fouls typically occur around frame 75
* Use ``--start_frame`` and ``--end_frame`` to focus on the incident
* Adjust ``--fps`` to control temporal resolution

4. Launch the Interface
----------------------

Use the interactive VARS interface:

.. code-block:: bash

   cd "VARS interface"
   python main.py

This will launch a GUI application where you can:

* Load video clips from multiple camera angles
* View real-time predictions for foul type and severity
* Compare model predictions with ground truth annotations
* Visualize attention weights across different views

Understanding the Output
-----------------------

The VARS model produces two main predictions:

**1. Action Classification (8 classes):**

* Tackling
* Standing tackling  
* High leg
* Holding
* Pushing
* Elbowing
* Challenge
* Dive

**2. Offence & Severity (4 classes):**

* No Offence
* Offence + No card
* Offence + Yellow card  
* Offence + Red card

Example Output Format
--------------------

Model predictions are saved in JSON format:

.. code-block:: json

   {
       "Actions": {
           "0": {
               "Action class": "High leg",
               "Offence": "Offence", 
               "Severity": "3.0"
           },
           "1": {
               "Action class": "Standing tackling",
               "Offence": "Offence",
               "Severity": "1.0"
           }
       }
   }

Sample Training Script
---------------------

Here's a complete training example:

.. code-block:: python

   import torch
   from model import MVNetwork
   from dataset import MultiViewDataset
   from train import trainer
   
   # Set device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Create model
   model = MVNetwork(
       net_name="mvit_v2_s",
       agr_type="attention"
   ).to(device)
   
   # Create dataset
   dataset = MultiViewDataset(
       path="path/to/dataset",
       start=63,
       end=87, 
       fps=17,
       split='Train',
       num_views=2
   )
   
   # Create data loader
   train_loader = torch.utils.data.DataLoader(
       dataset,
       batch_size=8,
       shuffle=True,
       num_workers=4
   )
   
   # Train model
   trained_model = trainer(
       model=model,
       train_loader=train_loader,
       epochs=15,
       device=device
   )

Performance Expectations
-----------------------

With the default configuration, you can expect:

**Training Time:**
* ~2-3 hours on RTX 3080 (full dataset)
* ~30 minutes on RTX 4090 (full dataset)

**Memory Usage:**
* ~8GB GPU memory (batch size 8)
* ~12GB RAM

**Accuracy:**
* Action classification: ~75-80% balanced accuracy
* Offence/Severity: ~80-85% balanced accuracy

Common Commands
--------------

**View model architecture:**

.. code-block:: bash

   python -c "from model import MVNetwork; print(MVNetwork())"

**Check dataset statistics:**

.. code-block:: bash

   python -c "from dataset import MultiViewDataset; d=MultiViewDataset('path', 0, 125, 25, 'Train', 2); print(len(d))"

**Evaluate on test set:**

.. code-block:: bash

   python main.py --only_test 1 --path_to_model_weights "model.pth.tar"

Next Steps
---------

Now that you have VARS running:

1. Explore the :doc:`dataset` to understand the data structure
2. Learn about :doc:`model_training` for advanced configurations  
3. Try the :doc:`interface_usage` for interactive analysis
4. Check :doc:`api/model` for programmatic usage
5. See :doc:`examples` for more use cases

Troubleshooting
--------------

**Model won't train:**
- Check GPU memory usage: ``nvidia-smi``
- Reduce batch size if out of memory
- Verify dataset path is correct

**Interface won't start:**
- Ensure PyQt5 is installed: ``pip install PyQt5``
- Check model weights are in correct location
- Verify all dependencies are installed

**Poor performance:**
- Train for more epochs
- Try different backbone models (``--pre_model``)
- Experiment with different aggregation methods (``--pooling_type``)
