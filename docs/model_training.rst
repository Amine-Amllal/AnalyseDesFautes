Model Training
==============

This guide covers the complete process of training VARS models for football foul detection, including data preparation, training configuration, and performance optimization.

Overview
--------

The VARS training pipeline supports multiple backbone architectures and aggregation methods. The system is designed for multi-view video analysis with temporal aggregation across camera angles.

**Supported Configurations:**

- **Backbones**: MViT_V2_S, R3D_18, R(2+1)D_18, MC3_18, S3D
- **Aggregation**: Max pooling, Mean pooling, Attention mechanism
- **Input Resolution**: 224x224 pixels
- **Temporal Window**: 2-second clips at 25 FPS

Training Setup
--------------

Data Preparation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.dataset import MultiViewDataset
   from torch.utils.data import DataLoader
   import torchvision.transforms as transforms
   
   # Define transformations
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])
   
   # Create datasets
   train_dataset = MultiViewDataset(
       path='/path/to/dataset',
       split='train',
       transform=transform,
       num_views=5,
       fps=25.0
   )
   
   val_dataset = MultiViewDataset(
       path='/path/to/dataset',
       split='valid',
       transform=transform,
       num_views=5,
       fps=25.0
   )
   
   # Create data loaders
   train_loader = DataLoader(
       train_dataset,
       batch_size=8,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )
   
   val_loader = DataLoader(
       val_dataset,
       batch_size=8,
       shuffle=False,
       num_workers=4,
       pin_memory=True
   )

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.model import MVNetwork
   import torch.nn as nn
   import torch.optim as optim
   
   # Initialize model
   model = MVNetwork(
       net_name="mvit_v2_s",
       agr_type="attention",
       num_classes_action=8,
       num_classes_severity=4
   )
   
   # Move to GPU if available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   
   # Define loss functions
   criterion_action = nn.CrossEntropyLoss()
   criterion_severity = nn.CrossEntropyLoss()
   
   # Setup optimizer
   optimizer = optim.Adam(
       model.parameters(),
       lr=1e-4,
       weight_decay=1e-4
   )
   
   # Learning rate scheduler
   scheduler = optim.lr_scheduler.StepLR(
       optimizer,
       step_size=10,
       gamma=0.5
   )

Training Loop
~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from tqdm import tqdm
   
   def train_epoch(model, train_loader, optimizer, criterion_action, criterion_severity, device):
       model.train()
       total_loss = 0.0
       action_correct = 0
       severity_correct = 0
       total_samples = 0
       
       for batch_idx, (videos, action_labels, severity_labels) in enumerate(tqdm(train_loader)):
           # Move data to device
           videos = videos.to(device)
           action_labels = action_labels.to(device)
           severity_labels = severity_labels.to(device)
           
           # Zero gradients
           optimizer.zero_grad()
           
           # Forward pass
           action_outputs, severity_outputs = model(videos)
           
           # Compute losses
           loss_action = criterion_action(action_outputs, action_labels)
           loss_severity = criterion_severity(severity_outputs, severity_labels)
           total_loss_batch = loss_action + loss_severity
           
           # Backward pass
           total_loss_batch.backward()
           optimizer.step()
           
           # Calculate accuracy
           _, action_predicted = torch.max(action_outputs.data, 1)
           _, severity_predicted = torch.max(severity_outputs.data, 1)
           
           total_samples += action_labels.size(0)
           action_correct += (action_predicted == action_labels).sum().item()
           severity_correct += (severity_predicted == severity_labels).sum().item()
           total_loss += total_loss_batch.item()
       
       # Calculate epoch metrics
       avg_loss = total_loss / len(train_loader)
       action_accuracy = 100 * action_correct / total_samples
       severity_accuracy = 100 * severity_correct / total_samples
       
       return avg_loss, action_accuracy, severity_accuracy
   
   def validate_epoch(model, val_loader, criterion_action, criterion_severity, device):
       model.eval()
       total_loss = 0.0
       action_correct = 0
       severity_correct = 0
       total_samples = 0
       
       with torch.no_grad():
           for videos, action_labels, severity_labels in val_loader:
               videos = videos.to(device)
               action_labels = action_labels.to(device)
               severity_labels = severity_labels.to(device)
               
               # Forward pass
               action_outputs, severity_outputs = model(videos)
               
               # Compute losses
               loss_action = criterion_action(action_outputs, action_labels)
               loss_severity = criterion_severity(severity_outputs, severity_labels)
               total_loss += (loss_action + loss_severity).item()
               
               # Calculate accuracy
               _, action_predicted = torch.max(action_outputs.data, 1)
               _, severity_predicted = torch.max(severity_outputs.data, 1)
               
               total_samples += action_labels.size(0)
               action_correct += (action_predicted == action_labels).sum().item()
               severity_correct += (severity_predicted == severity_labels).sum().item()
       
       avg_loss = total_loss / len(val_loader)
       action_accuracy = 100 * action_correct / total_samples
       severity_accuracy = 100 * severity_correct / total_samples
       
       return avg_loss, action_accuracy, severity_accuracy

Main Training Script
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Training configuration
   num_epochs = 50
   best_val_accuracy = 0.0
   patience = 10
   patience_counter = 0
   
   # Training history
   train_losses = []
   val_losses = []
   train_action_accs = []
   train_severity_accs = []
   val_action_accs = []
   val_severity_accs = []
   
   for epoch in range(num_epochs):
       print(f'Epoch [{epoch+1}/{num_epochs}]')
       
       # Training phase
       train_loss, train_action_acc, train_severity_acc = train_epoch(
           model, train_loader, optimizer, criterion_action, criterion_severity, device
       )
       
       # Validation phase
       val_loss, val_action_acc, val_severity_acc = validate_epoch(
           model, val_loader, criterion_action, criterion_severity, device
       )
       
       # Update learning rate
       scheduler.step()
       
       # Save metrics
       train_losses.append(train_loss)
       val_losses.append(val_loss)
       train_action_accs.append(train_action_acc)
       train_severity_accs.append(train_severity_acc)
       val_action_accs.append(val_action_acc)
       val_severity_accs.append(val_severity_acc)
       
       # Print epoch results
       print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
       print(f'Train Action Acc: {train_action_acc:.2f}%, Train Severity Acc: {train_severity_acc:.2f}%')
       print(f'Val Action Acc: {val_action_acc:.2f}%, Val Severity Acc: {val_severity_acc:.2f}%')
       
       # Save best model
       current_accuracy = (val_action_acc + val_severity_acc) / 2
       if current_accuracy > best_val_accuracy:
           best_val_accuracy = current_accuracy
           patience_counter = 0
           
           # Save checkpoint
           torch.save({
               'epoch': epoch,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'best_accuracy': best_val_accuracy,
               'train_losses': train_losses,
               'val_losses': val_losses,
               'train_action_accs': train_action_accs,
               'train_severity_accs': train_severity_accs,
               'val_action_accs': val_action_accs,
               'val_severity_accs': val_severity_accs,
           }, f'best_model_epoch_{epoch}.pth.tar')
           
           print(f'New best model saved! Accuracy: {best_val_accuracy:.2f}%')
       else:
           patience_counter += 1
       
       # Early stopping
       if patience_counter >= patience:
           print(f'Early stopping triggered after {patience} epochs without improvement')
           break
       
       print('-' * 60)

Advanced Training Techniques
----------------------------

Data Augmentation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torchvision.transforms as transforms
   from torchvision.transforms import functional as F
   
   class VideoAugmentation:
       def __init__(self, p=0.5):
           self.p = p
       
       def __call__(self, video):
           # Random horizontal flip
           if torch.rand(1) < self.p:
               video = F.hflip(video)
           
           # Random rotation
           if torch.rand(1) < self.p:
               angle = torch.randint(-15, 16, (1,)).item()
               video = F.rotate(video, angle)
           
           # Random brightness/contrast
           if torch.rand(1) < self.p:
               brightness = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
               video = F.adjust_brightness(video, brightness)
           
           return video
   
   # Apply augmentation to training data
   train_transform = transforms.Compose([
       VideoAugmentation(p=0.5),
       transforms.Resize((224, 224)),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.cuda.amp import GradScaler, autocast
   
   # Initialize gradient scaler
   scaler = GradScaler()
   
   def train_epoch_amp(model, train_loader, optimizer, criterion_action, criterion_severity, device):
       model.train()
       total_loss = 0.0
       
       for videos, action_labels, severity_labels in train_loader:
           videos = videos.to(device)
           action_labels = action_labels.to(device)
           severity_labels = severity_labels.to(device)
           
           optimizer.zero_grad()
           
           # Forward pass with autocast
           with autocast():
               action_outputs, severity_outputs = model(videos)
               loss_action = criterion_action(action_outputs, action_labels)
               loss_severity = criterion_severity(severity_outputs, severity_labels)
               total_loss_batch = loss_action + loss_severity
           
           # Backward pass with gradient scaling
           scaler.scale(total_loss_batch).backward()
           scaler.step(optimizer)
           scaler.update()
           
           total_loss += total_loss_batch.item()
       
       return total_loss / len(train_loader)

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from torch.nn.parallel import DataParallel
   
   # Check for multiple GPUs
   if torch.cuda.device_count() > 1:
       print(f"Using {torch.cuda.device_count()} GPUs")
       model = DataParallel(model)
   
   model = model.to(device)
   
   # Adjust batch size for multiple GPUs
   effective_batch_size = batch_size * torch.cuda.device_count()

Hyperparameter Tuning
----------------------

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Cosine annealing scheduler
   scheduler = optim.lr_scheduler.CosineAnnealingLR(
       optimizer,
       T_max=num_epochs,
       eta_min=1e-6
   )
   
   # Reduce on plateau
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(
       optimizer,
       mode='max',
       factor=0.5,
       patience=5,
       verbose=True
   )

Optimizer Selection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Adam optimizer (default)
   optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
   
   # AdamW optimizer (recommended)
   optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
   
   # SGD with momentum
   optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

Loss Function Weighting
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Weighted loss for imbalanced classes
   action_weights = torch.tensor([0.5, 2.0, 1.5, 1.0, 1.2, 3.0, 2.5, 2.0])
   severity_weights = torch.tensor([0.3, 1.0, 2.0, 3.0])
   
   criterion_action = nn.CrossEntropyLoss(weight=action_weights.to(device))
   criterion_severity = nn.CrossEntropyLoss(weight=severity_weights.to(device))

Performance Monitoring
----------------------

Metrics Tracking
~~~~~~~~~~~~~~~~

.. code-block:: python

   import wandb
   from sklearn.metrics import classification_report, confusion_matrix
   
   # Initialize Weights & Biases
   wandb.init(
       project="vars-training",
       config={
           "learning_rate": 1e-4,
           "batch_size": 8,
           "epochs": 50,
           "model": "mvit_v2_s",
           "aggregation": "attention"
       }
   )
   
   # Log metrics during training
   wandb.log({
       "epoch": epoch,
       "train_loss": train_loss,
       "val_loss": val_loss,
       "train_action_acc": train_action_acc,
       "val_action_acc": val_action_acc,
       "learning_rate": optimizer.param_groups[0]['lr']
   })

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   def plot_training_history(train_losses, val_losses, train_accs, val_accs):
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
       
       # Plot losses
       ax1.plot(train_losses, label='Train Loss')
       ax1.plot(val_losses, label='Validation Loss')
       ax1.set_title('Training and Validation Loss')
       ax1.set_xlabel('Epoch')
       ax1.set_ylabel('Loss')
       ax1.legend()
       
       # Plot accuracies
       ax2.plot(train_accs, label='Train Accuracy')
       ax2.plot(val_accs, label='Validation Accuracy')
       ax2.set_title('Training and Validation Accuracy')
       ax2.set_xlabel('Epoch')
       ax2.set_ylabel('Accuracy (%)')
       ax2.legend()
       
       plt.tight_layout()
       plt.savefig('training_history.png')
       plt.show()

Model Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   def detailed_evaluation(model, test_loader, device):
       model.eval()
       all_action_preds = []
       all_severity_preds = []
       all_action_labels = []
       all_severity_labels = []
       
       with torch.no_grad():
           for videos, action_labels, severity_labels in test_loader:
               videos = videos.to(device)
               action_outputs, severity_outputs = model(videos)
               
               _, action_preds = torch.max(action_outputs, 1)
               _, severity_preds = torch.max(severity_outputs, 1)
               
               all_action_preds.extend(action_preds.cpu().numpy())
               all_severity_preds.extend(severity_preds.cpu().numpy())
               all_action_labels.extend(action_labels.cpu().numpy())
               all_severity_labels.extend(severity_labels.cpu().numpy())
       
       # Generate classification reports
       action_report = classification_report(all_action_labels, all_action_preds)
       severity_report = classification_report(all_severity_labels, all_severity_preds)
       
       print("Action Classification Report:")
       print(action_report)
       print("\nSeverity Classification Report:")
       print(severity_report)
       
       return all_action_preds, all_severity_preds, all_action_labels, all_severity_labels

Troubleshooting
---------------

Common Training Issues
~~~~~~~~~~~~~~~~~~~~~~

**Out of Memory Errors:**

.. code-block:: python

   # Reduce batch size
   batch_size = 4  # Instead of 8
   
   # Use gradient accumulation
   accumulation_steps = 2
   optimizer.zero_grad()
   
   for i, (videos, labels) in enumerate(train_loader):
       outputs = model(videos)
       loss = criterion(outputs, labels) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

**Slow Training:**

.. code-block:: python

   # Optimize data loading
   train_loader = DataLoader(
       dataset,
       batch_size=batch_size,
       num_workers=8,  # Increase workers
       pin_memory=True,  # For GPU training
       persistent_workers=True  # Keep workers alive
   )

**Poor Convergence:**

- Lower learning rate
- Add batch normalization
- Use learning rate warmup
- Check data preprocessing

**Overfitting:**

- Add dropout layers
- Increase data augmentation
- Use early stopping
- Reduce model complexity

Best Practices
--------------

1. **Start Small**: Begin with a smaller model and dataset subset
2. **Monitor Closely**: Track both training and validation metrics
3. **Save Frequently**: Create checkpoints at regular intervals
4. **Use Version Control**: Track code changes and hyperparameters
5. **Document Everything**: Record training configurations and results
6. **Validate Thoroughly**: Test on held-out data before deployment
7. **Consider Resources**: Balance training time with available compute
8. **Experiment Systematically**: Change one parameter at a time
