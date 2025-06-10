Dataset Documentation
====================

The SoccerNet-MVFoul dataset is a comprehensive multi-view video dataset specifically designed for automatic foul recognition in football/soccer.

Dataset Overview
---------------

The SoccerNet-MVFoul dataset contains **3,901 football action clips** captured from multiple camera angles, each professionally annotated with detailed foul characteristics.

.. image:: ../images/dataset_example.png
   :alt: Dataset Example
   :align: center
   :width: 600px

Key Statistics
--------------

==================  ==========
Split               Actions
==================  ==========
Training Set        2,916
Validation Set      411  
Test Set            301
Challenge Set       273*
==================  ==========

*Challenge set has no public annotations

Dataset Structure
----------------

Each action in the dataset consists of:

* **Multiple camera views** (minimum 2, up to 4 views per action)
* **Live action footage** from the main broadcast cameras
* **Replay footage** from alternative angles
* **Professional annotations** for 10 different foul properties

Directory Organization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   dataset/
   ├── Train/
   │   ├── annotations.json
   │   └── action_X/
   │       ├── clip_0.mp4  # Main camera view
   │       ├── clip_1.mp4  # Alternative view 1
   │       ├── clip_2.mp4  # Alternative view 2
   │       └── ...
   ├── Valid/
   │   ├── annotations.json
   │   └── action_X/
   │       └── ...
   ├── Test/
   │   ├── annotations.json  
   │   └── action_X/
   │       └── ...
   └── Challenge/
       └── action_X/
           └── ...

Video Specifications
-------------------

**Technical Details:**

* **Resolution**: 480p (default) or 720p (optional)
* **Frame Rate**: 25 FPS
* **Duration**: ~5 seconds per clip (125 frames)
* **Format**: MP4 with H.264 encoding
* **Audio**: No audio track included

**Temporal Structure:**

* **Frame 0-62**: Pre-incident context
* **Frame 63-87**: Critical incident window (typical foul occurrence)
* **Frame 88-124**: Post-incident context

Annotation Schema
----------------

Each action is annotated with **10 properties** by professional referees:

Primary Classification Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Action Class (8 categories):**

============================  ===========  ===========================================
Class                         Code         Description
============================  ===========  ===========================================
Tackling                      0            Standard sliding tackle
Standing tackling             1            Tackle made while standing
High leg                      2            Dangerous high foot/leg contact
Holding                       3            Grabbing or holding an opponent
Pushing                       4            Pushing with hands or body
Elbowing                      5            Contact with elbow
Challenge                     6            General physical challenge
Dive                          7            Simulation/diving
============================  ===========  ===========================================

**2. Offence & Severity (4 categories):**

============================  ===========  ===========================================
Category                      Code         Description  
============================  ===========  ===========================================
No Offence                    0            Legal action, no foul
Offence + No card             1            Foul committed, no disciplinary action
Offence + Yellow card         2            Cautionable offence
Offence + Red card            3            Sending-off offence
============================  ===========  ===========================================

Additional Annotation Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset also includes annotations for:

* **UrlLocal**: Local path to the match video
* **Contact**: Contact type (With contact / Without contact)
* **Bodypart**: Body part involved (Upper body / Under body)
* **Upper body part**: Specific upper body part if applicable (Use of shoulder, Use of arms, etc.)
* **Multiple fouls**: Whether multiple fouls occurred
* **Try to play**: Whether the player tried to play the ball (Yes/No)
* **Touch ball**: Whether the ball was touched (Yes/No/Maybe)
* **Handball**: Handball status (No handball / Handball)
* **Handball offence**: Handball offence classification if applicable

Annotation Format
----------------

Annotations are stored in JSON format:

.. code-block:: json

   {
       "Set": "train",
       "Number of actions": 2916,
       "Actions": {
           "0": {
               "UrlLocal": "england_epl\\2014-2015\\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
               "Offence": "Offence",
               "Contact": "With contact",
               "Bodypart": "Upper body",
               "Upper body part": "Use of shoulder",
               "Action class": "Challenge",
               "Severity": "1.0",
               "Multiple fouls": "",
               "Try to play": "",
               "Touch ball": "",
               "Handball": "No handball",
               "Handball offence": "",
               "Clips": [
                   {
                       "Url": "Dataset/Train/action_0/clip_0",
                       "Camera type": "Main camera center",
                       "Timestamp": 1730826,
                       "Replay speed": 1.0
                   },
                   {
                       "Url": "Dataset/Train/action_0/clip_1",
                       "Camera type": "Close-up player or field referee",
                       "Timestamp": 1744173,
                       "Replay speed": 1.8
                   }
               ]
           }
       }
   }

Data Loading
-----------

Python API for Loading Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataset import MultiViewDataset
   import torch
   
   # Create dataset instance
   dataset = MultiViewDataset(
       path="path/to/dataset",
       start=63,        # Start frame
       end=87,          # End frame  
       fps=17,          # Target FPS
       split='Train',   # Data split
       num_views=2      # Number of views to load
   )
   
   # Create data loader
   data_loader = torch.utils.data.DataLoader(
       dataset,
       batch_size=8,
       shuffle=True,
       num_workers=4
   )
   
   # Iterate through data
   for offence_labels, action_labels, videos, action_ids in data_loader:
       # offence_labels: [batch_size, 4] - one-hot encoded
       # action_labels: [batch_size, 8] - one-hot encoded  
       # videos: [batch_size, views, channels, frames, height, width]
       # action_ids: List of action identifiers
       pass

Dataset Configuration
~~~~~~~~~~~~~~~~~~~

**Key Parameters:**

* ``start_frame`` / ``end_frame``: Define temporal window
* ``fps``: Resample frame rate (original is 25 FPS)
* ``num_views``: Number of camera views to load
* ``transform``: Data augmentation transformations

Example configurations:

.. code-block:: python

   # Full clip analysis
   dataset_full = MultiViewDataset(
       path="dataset", start=0, end=125, fps=25, 
       split='Train', num_views=5
   )
   
   # Focus on incident
   dataset_incident = MultiViewDataset(
       path="dataset", start=63, end=87, fps=17,
       split='Train', num_views=2  
   )

Data Augmentation
----------------

The dataset supports various augmentation techniques:

.. code-block:: python

   import torchvision.transforms as transforms
   
   # Define augmentation pipeline
   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.ColorJitter(
           brightness=0.2, 
           contrast=0.2, 
           saturation=0.2, 
           hue=0.1
       ),
       transforms.RandomAffine(
           degrees=10,
           translate=(0.1, 0.1),
           scale=(0.9, 1.1)
       )
   ])
   
   # Apply to dataset
   dataset = MultiViewDataset(
       path="dataset",
       split='Train', 
       transform=transform,
       # ... other parameters
   )

Quality Assurance
----------------

**Annotation Quality:**

* All annotations performed by professional referees
* 6+ years of refereeing experience  
* 300+ official games officiated
* Inter-annotator agreement validation performed

**Data Quality Checks:**

* Video integrity verification
* Temporal alignment across views
* Resolution and quality standards
* Duplicate detection and removal

Usage Examples
-------------

**Basic Data Exploration:**

.. code-block:: python

   from collections import Counter
   import json
   
   # Load annotations
   with open('dataset/Train/annotations.json', 'r') as f:
       annotations = json.load(f)
   
   # Analyze class distribution
   action_classes = [
       annotations['Actions'][action]['Action class'] 
       for action in annotations['Actions']
   ]
   
   print("Action class distribution:")
   print(Counter(action_classes))

**Custom Data Filtering:**

.. code-block:: python

   def filter_high_quality_clips(annotations):
       """Filter clips with high visibility and quality."""
       filtered = {}
       for action_id, action_data in annotations['Actions'].items():
           if (action_data.get('Camera quality', '') == 'High' and 
               action_data.get('Visibility', '') == 'Clear'):
               filtered[action_id] = action_data
       return filtered

**Multi-View Analysis:**

.. code-block:: python

   # Analyze number of views per action
   view_counts = [
       len(annotations['Actions'][action]['Clips'])
       for action in annotations['Actions']
   ]
   
   print(f"Average views per action: {np.mean(view_counts):.2f}")
   print(f"Max views: {max(view_counts)}")
   print(f"Min views: {min(view_counts)}")

Data Download & Setup
--------------------

**Step 1: Get Access**

1. Fill out the NDA form: `SoccerNet NDA <https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform>`_
2. Receive password via email

**Step 2: Download Data**

.. code-block:: python

   from SoccerNet.Downloader import SoccerNetDownloader as SNdl
   
   mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
   mySNdl.downloadDataTask(
       task="mvfouls", 
       split=["train","valid","test","challenge"], 
       password="your_password"
   )

**Step 3: Extract and Organize**

.. code-block:: bash

   # Extract downloaded archives
   unzip train.zip
   unzip valid.zip  
   unzip test.zip
   unzip challenge.zip
   
   # Organize directory structure
   mkdir dataset
   mv Train Valid Test Challenge dataset/

Best Practices
-------------

**For Training:**

* Use ``num_views=2`` for training (memory efficiency)
* Apply data augmentation to increase dataset variety
* Focus on frames 63-87 for incident detection
* Use balanced sampling for class imbalance

**For Evaluation:**

* Use ``num_views=5`` for comprehensive evaluation
* Process full clips (frames 0-125) for context
* Disable data augmentation for consistent results
* Report metrics on both validation and test sets

**For Analysis:**

* Examine attention weights to understand model focus
* Compare performance across different camera angles
* Analyze failure cases by foul type and severity
* Consider temporal dynamics in decision making

Limitations & Considerations
---------------------------

* **Subjective nature**: Foul decisions can be subjective even among referees
* **Camera angles**: Not all incidents are equally visible from all angles
* **Temporal alignment**: Slight misalignment possible between different views
* **Quality variation**: Video quality varies across different matches and cameras
* **Context limitation**: 5-second clips may not capture full incident context

For more detailed information about using the dataset in your research, see the :doc:`model_training` and :doc:`evaluation_metrics` sections.
