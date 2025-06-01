VARS - Video Assistant Referee System
=====================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/pytorch-1.12+-orange.svg
   :target: https://pytorch.org/
   :alt: PyTorch 1.12+

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

The Video Assistant Referee (VAR) has revolutionized association football, enabling referees to review incidents on the pitch, making informed decisions, and ensuring fairness. However, due to the lack of referees in many countries and the high cost of the VAR infrastructure, only professional leagues can benefit from it.

**VARS** (Video Assistant Referee System) represents a first step towards a fully automated video assistant referee system that could support or replace the current VAR technology.

🚀 **Key Features**
------------------

* **SoccerNet-MVFoul Dataset**: Multi-view video dataset with 3,901 foul incidents captured by multiple cameras
* **Multi-View Video Architecture**: Advanced deep learning system for classifying foul types and severity
* **Interactive Interface**: User-friendly GUI for real-time foul analysis and visualization
* **Professional Annotations**: All incidents annotated by professional referees with 6+ years experience

📋 **What's Included**
--------------------

This repository contains three main components:

1. **SoccerNet-MVFoul Dataset**: A comprehensive multi-view video dataset containing clips of fouls captured by multiple cameras, annotated with 10 different properties
2. **VARS Model**: A multi-camera video recognition system for classifying foul types and severity using state-of-the-art deep learning techniques
3. **VARS Interface**: An interactive application that displays ground truth actions and model predictions with confidence scores

🎯 **Quick Start**
-----------------

.. code-block:: bash

   # Create virtual environment
   conda create -n vars python=3.9
   conda activate vars
   
   # Install dependencies
   pip install SoccerNet
   pip install -r requirements.txt
   pip install pyav
   
   # Train the model
   python main.py --path "path/to/dataset"

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   quickstart
   dataset
   model_training
   interface_usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/model
   api/dataset
   api/evaluation
   api/interface

.. toctree::
   :maxdepth: 2
   :caption: Technical Details
   :hidden:

   architecture
   evaluation_metrics
   data_format

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   examples
   troubleshooting
   contributing
   changelog

📊 **Dataset Overview**
----------------------

The SoccerNet-MVFoul dataset consists of:

* **3,901 total actions** across multiple camera angles
* **Training set**: 2,916 actions
* **Validation set**: 411 actions  
* **Test set**: 301 actions
* **Challenge set**: 273 actions (without annotations)

Each action is annotated with **10 properties** describing foul characteristics from a referee perspective, including foul type, severity, and context.

🏗️ **Architecture**
------------------

Our VARS employs a three-stage pipeline:

1. **Encoder (E)**: Extracts features from each camera view independently
2. **Aggregator (A)**: Combines multi-view information using attention mechanisms
3. **Classifier (C)**: Determines foul properties through specialized classification heads

🎖️ **Performance**
-----------------

The system achieves state-of-the-art performance on multi-view foul recognition:

* **Action Classification**: 8 classes (Tackling, Standing tackling, High leg, etc.)
* **Offence & Severity**: 4 classes (No Offence, Offence + No card, Yellow card, Red card)
* **Evaluation Metric**: Balanced accuracy across both tasks

👥 **Authors**
-------------

* **AMLLAL Amine**
* **AKEBLI FatimaEzzahrae** 
* **ELHAKIOUI Asmae**

📄 **License**
-------------

This project is licensed under the MIT License - see the LICENSE file for details.

🔗 **Related Links**
------------------

* `SoccerNet Official Website <https://www.soccer-net.org/>`_
* `PyTorch Documentation <https://pytorch.org/docs/stable/index.html>`_
* `Original Paper <#>`_ (Add link when available)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

