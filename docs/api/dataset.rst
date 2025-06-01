Dataset API Reference
====================

This section provides detailed API documentation for the VARS dataset components.

MultiViewDataset Class
----------------------

The main dataset class for loading and processing multi-view video data.

.. autoclass:: dataset.MultiViewDataset
   :members:
   :inherited-members:

Class Definition
~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiViewDataset(Dataset):
       """
       PyTorch Dataset for multi-view football foul recognition.
       
       Loads video clips from multiple camera angles along with 
       corresponding annotations for foul classification tasks.
       
       Args:
           path (str): Root path to dataset directory
           start (int): Starting frame index (0-124)
           end (int): Ending frame index (0-124)  
           fps (int): Target frames per second for resampling
           split (str): Dataset split ("Train", "Valid", "Test", "Chall")
           num_views (int): Number of camera views to load (1-5)
           transform (callable, optional): Video augmentation transforms
           transform_model (callable, optional): Model preprocessing transforms
       """

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~

==================  ===========  ==========================================
Parameter           Type         Description
==================  ===========  ==========================================
``path``            str          Path to dataset root directory
``start``           int          Start frame (0-124, typically 63)
``end``             int          End frame (0-124, typically 87)  
``fps``             int          Target FPS (original is 25)
``split``           str          Data split identifier
``num_views``       int          Number of views to load per sample
``transform``       callable     Optional video augmentation
``transform_model`` callable     Preprocessing for model input
==================  ===========  ==========================================

**Split Options:**

* ``"Train"``: Training set (2,916 actions)
* ``"Valid"``: Validation set (411 actions)
* ``"Test"``: Test set (301 actions)  
* ``"Chall"``: Challenge set (273 actions, no annotations)

Methods
~~~~~~~

**__getitem__(index)**

.. code-block:: python

   def __getitem__(self, index):
       """
       Get a single sample from the dataset.
       
       Args:
           index (int): Sample index
           
       Returns:
           tuple: (offence_labels, action_labels, videos, action_id)
               - offence_labels: One-hot tensor [4] for offence/severity
               - action_labels: One-hot tensor [8] for action type
               - videos: Video tensor [V, C, T, H, W]
               - action_id: String identifier for the action
       """

**__len__()**

.. code-block:: python

   def __len__(self):
       """Return the total number of samples in the dataset."""
       return len(self.clips)

**getDistribution()**

.. code-block:: python

   def getDistribution(self):
       """
       Get class distribution statistics.
       
       Returns:
           tuple: (action_distribution, offence_distribution)
               - action_distribution: Tensor [8] with class frequencies
               - offence_distribution: Tensor [4] with class frequencies
       """

**getWeights()**

.. code-block:: python

   def getWeights(self):
       """
       Get class weights for balanced training.
       
       Returns:
           tuple: (action_weights, offence_weights)
               - action_weights: Tensor [8] with inverse class frequencies
               - offence_weights: Tensor [4] with inverse class frequencies
       """

Usage Examples
~~~~~~~~~~~~~

**Basic Dataset Creation:**

.. code-block:: python

   from dataset import MultiViewDataset
   import torchvision.transforms as transforms
   
   # Model preprocessing transforms
   transform_model = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])
   
   # Create dataset
   dataset = MultiViewDataset(
       path="path/to/dataset",
       start=63,                    # Focus on incident frames
       end=87,
       fps=17,                      # Downsample from 25 FPS
       split='Train',
       num_views=2,                 # Load 2 camera views
       transform_model=transform_model
   )
   
   print(f"Dataset size: {len(dataset)}")

**With Data Augmentation:**

.. code-block:: python

   # Training augmentation transforms
   transform_aug = transforms.Compose([
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
   
   # Training dataset with augmentation
   train_dataset = MultiViewDataset(
       path="dataset",
       start=0, end=125,           # Full clips for training
       fps=25,
       split='Train', 
       num_views=2,
       transform=transform_aug,     # Apply augmentation
       transform_model=transform_model
   )

**Accessing Samples:**

.. code-block:: python

   # Get a single sample
   offence_labels, action_labels, videos, action_id = dataset[0]
   
   print(f"Offence labels shape: {offence_labels.shape}")    # [4]
   print(f"Action labels shape: {action_labels.shape}")      # [8]
   print(f"Videos shape: {videos.shape}")                    # [2, 3, 25, 224, 224]
   print(f"Action ID: {action_id}")
   
   # Decode labels
   offence_class = torch.argmax(offence_labels).item()
   action_class = torch.argmax(action_labels).item()
   
   print(f"Offence class: {offence_class}")  # 0-3
   print(f"Action class: {action_class}")    # 0-7

Data Loading Functions
---------------------

**label2vectormerge()**

.. code-block:: python

   def label2vectormerge(folder_path, split, num_views):
       """
       Load and process annotation labels.
       
       Args:
           folder_path (str): Path to dataset directory
           split (str): Dataset split name
           num_views (int): Number of views per sample
           
       Returns:
           tuple: (labels_offence_severity, labels_action, 
                   distribution_offence, distribution_action,
                   not_taking, number_of_actions)
       """

**clips2vectormerge()**

.. code-block:: python

   def clips2vectormerge(folder_path, split, num_views, not_taking):
       """
       Load video clip paths.
       
       Args:
           folder_path (str): Path to dataset directory 
           split (str): Dataset split name
           num_views (int): Number of views per sample
           not_taking (list): List of actions to exclude
           
       Returns:
           list: List of clip paths for each action
       """

DataLoader Configuration
-----------------------

**Training DataLoader:**

.. code-block:: python

   from torch.utils.data import DataLoader
   
   train_loader = DataLoader(
       dataset=train_dataset,
       batch_size=8,
       shuffle=True,
       num_workers=4,
       pin_memory=True,
       drop_last=True           # For stable batch norm
   )

**Validation/Test DataLoader:**

.. code-block:: python

   val_loader = DataLoader(
       dataset=val_dataset,
       batch_size=1,            # Process one sample at a time
       shuffle=False,
       num_workers=2,
       pin_memory=True
   )

**Custom Collate Function:**

.. code-block:: python

   def custom_collate_fn(batch):
       """Custom collate function for variable number of views."""
       offence_labels = torch.stack([item[0] for item in batch])
       action_labels = torch.stack([item[1] for item in batch]) 
       videos = torch.stack([item[2] for item in batch])
       action_ids = [item[3] for item in batch]
       
       return offence_labels, action_labels, videos, action_ids
   
   loader = DataLoader(dataset, collate_fn=custom_collate_fn, ...)

Dataset Statistics
-----------------

**Class Distribution Analysis:**

.. code-block:: python

   def analyze_dataset(dataset):
       """Analyze dataset class distribution."""
       action_dist, offence_dist = dataset.getDistribution()
       action_weights, offence_weights = dataset.getWeights()
       
       # Action classes
       action_classes = [
           "Tackling", "Standing tackling", "High leg", "Holding",
           "Pushing", "Elbowing", "Challenge", "Dive"
       ]
       
       # Offence classes  
       offence_classes = [
           "No Offence", "Offence + No card", 
           "Offence + Yellow card", "Offence + Red card"
       ]
       
       print("Action Distribution:")
       for i, (cls, freq, weight) in enumerate(
           zip(action_classes, action_dist, action_weights)
       ):
           print(f"  {cls}: {freq:.3f} (weight: {weight:.3f})")
       
       print("\nOffence Distribution:")
       for i, (cls, freq, weight) in enumerate(
           zip(offence_classes, offence_dist, offence_weights)
       ):
           print(f"  {cls}: {freq:.3f} (weight: {weight:.3f})")

**Sample Analysis:**

.. code-block:: python

   def sample_analysis(dataset, num_samples=10):
       """Analyze random samples from dataset."""
       import random
       
       indices = random.sample(range(len(dataset)), num_samples)
       
       for idx in indices:
           offence_labels, action_labels, videos, action_id = dataset[idx]
           
           print(f"Sample {idx} (Action {action_id}):")
           print(f"  Video shape: {videos.shape}")
           print(f"  Action: {torch.argmax(action_labels).item()}")
           print(f"  Offence: {torch.argmax(offence_labels).item()}")

Data Transformations
-------------------

**Standard Preprocessing:**

.. code-block:: python

   def get_preprocessing_transforms():
       """Get standard preprocessing transforms."""
       return transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ConvertImageDtype(torch.float32),
           transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
           )
       ])

**Training Augmentations:**

.. code-block:: python

   def get_training_transforms():
       """Get training augmentation transforms."""
       return transforms.Compose([
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.ColorJitter(
               brightness=0.3,
               contrast=0.3,
               saturation=0.3,
               hue=0.1
           ),
           transforms.RandomAffine(
               degrees=15,
               translate=(0.1, 0.1),
               scale=(0.8, 1.2),
               shear=5
           ),
           transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
       ])

**Temporal Augmentations:**

.. code-block:: python

   class TemporalSubsampling:
       """Randomly subsample frames from video clips."""
       
       def __init__(self, target_frames=16):
           self.target_frames = target_frames
           
       def __call__(self, video):
           """
           Args:
               video: Tensor of shape [T, H, W, C]
           Returns:
               Subsampled video of shape [target_frames, H, W, C]
           """
           T = video.shape[0]
           if T <= self.target_frames:
               return video
           
           # Random temporal sampling
           indices = torch.linspace(0, T-1, self.target_frames).long()
           return video[indices]

Error Handling
-------------

**Dataset Validation:**

.. code-block:: python

   def validate_dataset(dataset_path, split):
       """Validate dataset integrity."""
       try:
           dataset = MultiViewDataset(
               path=dataset_path,
               start=0, end=125, fps=25,
               split=split, num_views=1
           )
           
           # Test loading first sample
           sample = dataset[0]
           print(f"Dataset {split} is valid. Size: {len(dataset)}")
           
       except FileNotFoundError:
           print(f"Dataset path not found: {dataset_path}")
       except json.JSONDecodeError:
           print(f"Invalid annotations file for split: {split}")
       except Exception as e:
           print(f"Dataset validation failed: {e}")

**Common Issues:**

.. code-block:: python

   def handle_common_issues():
       """Handle common dataset loading issues."""
       
       # Issue 1: Missing video files
       try:
           videos = dataset[idx][2]
       except RuntimeError as e:
           if "No such file" in str(e):
               print("Missing video file. Check dataset integrity.")
       
       # Issue 2: Corrupted video files
       try:
           videos = dataset[idx][2]
       except Exception as e:
           if "codec" in str(e).lower():
               print("Video codec issue. Install additional codecs.")
       
       # Issue 3: Memory issues
       try:
           loader = DataLoader(dataset, batch_size=32)
       except RuntimeError as e:
           if "out of memory" in str(e):
               print("Reduce batch_size or num_workers.")

Performance Optimization
-----------------------

**Fast Loading Configuration:**

.. code-block:: python

   # Optimized for speed
   fast_dataset = MultiViewDataset(
       path="dataset",
       start=70, end=80,           # Minimal frames
       fps=12,                     # Lower FPS
       split='Train',
       num_views=1,                # Single view
       transform=None              # No augmentation
   )
   
   fast_loader = DataLoader(
       fast_dataset,
       batch_size=32,
       num_workers=8,              # More workers
       pin_memory=True,
       prefetch_factor=4
   )

**Memory-Efficient Configuration:**

.. code-block:: python

   # Optimized for memory
   memory_dataset = MultiViewDataset(
       path="dataset", 
       start=63, end=87,
       fps=17,
       split='Train',
       num_views=2,
       transform_model=transforms.Resize((112, 112))  # Smaller resolution
   )
   
   memory_loader = DataLoader(
       memory_dataset,
       batch_size=4,               # Smaller batches
       num_workers=2,
       pin_memory=False
   )

For more examples and advanced usage patterns, see the :doc:`../examples` and :doc:`../model_training` sections.
