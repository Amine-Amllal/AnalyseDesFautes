Model API Reference
==================

This section provides detailed API documentation for the VARS model components.

MVNetwork Class
--------------

The main neural network class that implements the complete VARS architecture.

.. autoclass:: model.MVNetwork
   :members:
   :inherited-members:

Class Definition
~~~~~~~~~~~~~~~

.. code-block:: python

   class MVNetwork(torch.nn.Module):
       """
       Multi-View Network for football foul recognition.
       
       This network processes multiple camera views of football incidents
       and classifies them into action types and offence severity levels.
       
       Args:
           net_name (str): Backbone architecture name. Options:
               - "r3d_18": 3D ResNet-18
               - "mc3_18": Mixed 3D convolutions  
               - "r2plus1d_18": R(2+1)D architecture
               - "s3d": Spatially separable 3D convolutions
               - "mvit_v2_s": Multi-scale Vision Transformer v2 (Small)
               - "mvit_v1_b": Multi-scale Vision Transformer v1 (Base)
           agr_type (str): Aggregation method. Options:
               - "max": Max pooling across views
               - "mean": Average pooling across views  
               - "attention": Learnable attention weights
           lifting_net (torch.nn.Module): Optional feature transformation network
           
       Returns:
           tuple: (offence_severity_logits, action_logits, attention_weights)
               - offence_severity_logits: [batch_size, 4] tensor
               - action_logits: [batch_size, 8] tensor
               - attention_weights: [batch_size, num_views] tensor (if attention aggregation)
       """

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~

==================  ===========  ==========================================
Parameter           Type         Description
==================  ===========  ==========================================
``net_name``        str          Backbone architecture identifier
``agr_type``        str          Multi-view aggregation method
``lifting_net``     nn.Module    Optional feature transformation layer
==================  ===========  ==========================================

**Supported Backbones:**

* ``"r3d_18"``: 3D ResNet-18 (512-dim features)
* ``"mc3_18"``: Mixed 3D CNN (512-dim features)
* ``"r2plus1d_18"``: R(2+1)D CNN (512-dim features)  
* ``"s3d"``: Spatially separable 3D CNN (400-dim features)
* ``"mvit_v2_s"``: MViT v2 Small (400-dim features) - **Recommended**
* ``"mvit_v1_b"``: MViT v1 Base (768-dim features)

Methods
~~~~~~~

**forward(mvimages)**

.. code-block:: python

   def forward(self, mvimages):
       """
       Forward pass through the network.
       
       Args:
           mvimages (torch.Tensor): Multi-view video tensor of shape
               [batch_size, num_views, channels, frames, height, width]
               
       Returns:
           tuple: (offence_logits, action_logits, attention_weights)
               - offence_logits: Predictions for offence/severity [B, 4]
               - action_logits: Predictions for action type [B, 8] 
               - attention_weights: View importance weights [B, V] (if applicable)
       """

Usage Examples
~~~~~~~~~~~~~

**Basic Usage:**

.. code-block:: python

   import torch
   from model import MVNetwork
   
   # Create model
   model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
   
   # Example input: 2 views, 3 channels, 25 frames, 224x224 resolution
   videos = torch.randn(4, 2, 3, 25, 224, 224)
   
   # Forward pass
   offence_logits, action_logits, attention = model(videos)
   
   print(f"Offence predictions: {offence_logits.shape}")  # [4, 4]
   print(f"Action predictions: {action_logits.shape}")    # [4, 8]
   print(f"Attention weights: {attention.shape}")         # [4, 2]

**With Custom Lifting Network:**

.. code-block:: python

   import torch.nn as nn
   
   # Define custom feature transformation
   lifting_net = nn.Sequential(
       nn.Linear(400, 512),
       nn.ReLU(),
       nn.Dropout(0.3),
       nn.Linear(512, 400)
   )
   
   # Create model with lifting network
   model = MVNetwork(
       net_name="mvit_v2_s",
       agr_type="attention", 
       lifting_net=lifting_net
   )

MVAggregate Class
----------------

Handles multi-view feature aggregation and classification.

.. code-block:: python

   class MVAggregate(nn.Module):
       """
       Multi-view feature aggregation and classification module.
       
       Combines features from multiple camera views and produces
       predictions for both action type and offence severity.
       """

Constructor
~~~~~~~~~~

.. code-block:: python

   def __init__(self, model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
       """
       Initialize MVAggregate module.
       
       Args:
           model (nn.Module): Backbone video encoder
           agr_type (str): Aggregation method ("max", "mean", "attention")
           feat_dim (int): Feature dimension from backbone
           lifting_net (nn.Module): Optional feature transformation
       """

Aggregation Methods
~~~~~~~~~~~~~~~~~~

**Max Pooling Aggregation:**

.. code-block:: python

   class ViewMaxAggregate(nn.Module):
       """Takes maximum activation across views."""
       
       def forward(self, mvimages):
           # Process each view
           features = self.extract_features(mvimages)  # [B, V, D]
           
           # Max pooling across views
           pooled_features = torch.max(features, dim=1)[0]  # [B, D]
           
           return pooled_features, features

**Average Pooling Aggregation:**

.. code-block:: python

   class ViewAvgAggregate(nn.Module):
       """Computes average activation across views."""
       
       def forward(self, mvimages):
           features = self.extract_features(mvimages)  # [B, V, D]
           pooled_features = torch.mean(features, dim=1)  # [B, D]
           return pooled_features, features

**Attention-Based Aggregation:**

.. code-block:: python

   class WeightedAggregate(nn.Module):
       """Uses learnable attention for view weighting."""
       
       def __init__(self, feat_dim):
           super().__init__()
           self.attention_weights = nn.Parameter(torch.randn(feat_dim, feat_dim))
           self.relu = nn.ReLU()
           
       def forward(self, mvimages):
           features = self.extract_features(mvimages)  # [B, V, D]
           
           # Compute attention scores
           attention_input = torch.matmul(features, self.attention_weights)
           attention_matrix = torch.bmm(attention_input, attention_input.transpose(1, 2))
           attention_weights = F.softmax(self.relu(attention_matrix).sum(dim=2), dim=1)
           
           # Apply attention weights
           weighted_features = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)
           
           return weighted_features, attention_weights

Utility Functions
----------------

**batch_tensor()**

.. code-block:: python

   def batch_tensor(tensor, dim=1, squeeze=False):
       """
       Reshape tensor to combine batch and view dimensions.
       
       Args:
           tensor (torch.Tensor): Input tensor [B, V, ...]
           dim (int): Dimension to batch along
           squeeze (bool): Whether to squeeze singleton dimensions
           
       Returns:
           torch.Tensor: Reshaped tensor [B*V, ...]
       """

**unbatch_tensor()**

.. code-block:: python

   def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
       """
       Reshape tensor to separate batch and view dimensions.
       
       Args:
           tensor (torch.Tensor): Input tensor [B*V, ...]
           batch_size (int): Original batch size
           dim (int): Dimension to unbatch along
           unsqueeze (bool): Whether to add singleton dimensions
           
       Returns:
           torch.Tensor: Reshaped tensor [B, V, ...]
       """

Model Configuration
------------------

**Default Configurations:**

.. code-block:: python

   # High accuracy configuration
   CONFIG_ACCURACY = {
       'net_name': 'mvit_v2_s',
       'agr_type': 'attention',
       'feat_dim': 400,
       'batch_size': 4,
       'num_views': 5
   }
   
   # Fast inference configuration  
   CONFIG_SPEED = {
       'net_name': 'r3d_18',
       'agr_type': 'max',
       'feat_dim': 512,
       'batch_size': 16,
       'num_views': 2
   }
   
   # Balanced configuration
   CONFIG_BALANCED = {
       'net_name': 'r2plus1d_18', 
       'agr_type': 'mean',
       'feat_dim': 512,
       'batch_size': 8,
       'num_views': 3
   }

Model Loading and Saving
-----------------------

**Save Trained Model:**

.. code-block:: python

   def save_model(model, optimizer, epoch, path):
       """Save model checkpoint."""
       checkpoint = {
           'epoch': epoch,
           'state_dict': model.state_dict(),
           'optimizer': optimizer.state_dict(),
           'config': {
               'net_name': model.net_name,
               'agr_type': model.agr_type,
               'feat_dim': model.feat_dim
           }
       }
       torch.save(checkpoint, path)

**Load Trained Model:**

.. code-block:: python

   def load_model(path, device='cuda'):
       """Load model from checkpoint."""
       checkpoint = torch.load(path, map_location=device)
       
       # Recreate model
       model = MVNetwork(
           net_name=checkpoint['config']['net_name'],
           agr_type=checkpoint['config']['agr_type'],
           feat_dim=checkpoint['config']['feat_dim']
       )
       
       # Load weights
       model.load_state_dict(checkpoint['state_dict'])
       model.eval()
       
       return model, checkpoint

**Model Information:**

.. code-block:: python

   def model_summary(model):
       """Print model architecture summary."""
       total_params = sum(p.numel() for p in model.parameters())
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       
       print(f"Model: {model.__class__.__name__}")
       print(f"Backbone: {model.net_name}")
       print(f"Aggregation: {model.agr_type}")
       print(f"Feature dim: {model.feat_dim}")
       print(f"Total parameters: {total_params:,}")
       print(f"Trainable parameters: {trainable_params:,}")

Error Handling
-------------

**Common Issues and Solutions:**

.. code-block:: python

   try:
       model = MVNetwork(net_name="mvit_v2_s")
       predictions = model(videos)
   except RuntimeError as e:
       if "out of memory" in str(e):
           print("GPU out of memory. Try reducing batch_size or num_views")
       elif "size mismatch" in str(e):
           print("Input tensor dimensions incorrect. Expected [B,V,C,T,H,W]")
       else:
           raise e

**Input Validation:**

.. code-block:: python

   def validate_input(videos):
       """Validate input tensor format."""
       assert videos.dim() == 6, f"Expected 6D tensor, got {videos.dim()}D"
       
       B, V, C, T, H, W = videos.shape
       assert C == 3, f"Expected 3 channels, got {C}"
       assert T >= 8, f"Minimum 8 frames required, got {T}"
       assert H >= 224 and W >= 224, f"Minimum 224x224 resolution required"
       assert V >= 1, f"At least 1 view required, got {V}"

Performance Optimization
-----------------------

**Memory Optimization:**

.. code-block:: python

   # Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Use mixed precision training
   from torch.cuda.amp import autocast
   
   with autocast():
       predictions = model(videos)

**Inference Optimization:**

.. code-block:: python

   # Disable gradients for inference
   model.eval()
   with torch.no_grad():
       predictions = model(videos)
   
   # Compile model for faster inference (PyTorch 2.0+)
   compiled_model = torch.compile(model)

For more detailed examples and advanced usage, see the :doc:`../examples` section.
