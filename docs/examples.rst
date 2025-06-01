Examples
========

This section provides practical examples and use cases for the VARS system, demonstrating how to use different components for football foul detection and analysis.

Quick Start Examples
-------------------

Basic Model Inference
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from src.model import MVNetwork
   from src.dataset import MultiViewDataset
   from torch.utils.data import DataLoader
   
   # Load pre-trained model
   model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
   checkpoint = torch.load('path/to/model.pth.tar', map_location='cpu')
   model.load_state_dict(checkpoint['state_dict'])
   model.eval()
   
   # Load a single video for inference
   dataset = MultiViewDataset(
       path='/path/to/dataset',
       split='test',
       fps=25.0,
       num_views=5
   )
   
   # Get a single sample
   video, action_label, severity_label = dataset[0]
   video = video.unsqueeze(0)  # Add batch dimension
   
   # Run inference
   with torch.no_grad():
       action_output, severity_output = model(video)
       
       # Get predictions
       action_pred = torch.argmax(action_output, dim=1).item()
       severity_pred = torch.argmax(severity_output, dim=1).item()
       
       print(f"Predicted action: {action_pred}")
       print(f"Predicted severity: {severity_pred}")

Batch Processing
~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple videos in batches
   test_loader = DataLoader(
       dataset,
       batch_size=8,
       shuffle=False,
       num_workers=4
   )
   
   predictions = []
   with torch.no_grad():
       for videos, action_labels, severity_labels in test_loader:
           action_outputs, severity_outputs = model(videos)
           
           # Get batch predictions
           action_preds = torch.argmax(action_outputs, dim=1)
           severity_preds = torch.argmax(severity_outputs, dim=1)
           
           # Store results
           for i in range(len(videos)):
               predictions.append({
                   'action_pred': action_preds[i].item(),
                   'severity_pred': severity_preds[i].item(),
                   'action_true': action_labels[i].item(),
                   'severity_true': severity_labels[i].item()
               })
   
   print(f"Processed {len(predictions)} videos")

Training Examples
----------------

Simple Training Loop
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   
   # Setup model and training components
   model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   
   # Loss functions and optimizer
   criterion_action = nn.CrossEntropyLoss()
   criterion_severity = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=1e-4)
   
   # Data loaders
   train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
   
   # Training loop
   num_epochs = 10
   for epoch in range(num_epochs):
       # Training phase
       model.train()
       train_loss = 0.0
       
       for videos, action_labels, severity_labels in train_loader:
           videos = videos.to(device)
           action_labels = action_labels.to(device)
           severity_labels = severity_labels.to(device)
           
           optimizer.zero_grad()
           
           # Forward pass
           action_outputs, severity_outputs = model(videos)
           
           # Calculate losses
           loss_action = criterion_action(action_outputs, action_labels)
           loss_severity = criterion_severity(severity_outputs, severity_labels)
           total_loss = loss_action + loss_severity
           
           # Backward pass
           total_loss.backward()
           optimizer.step()
           
           train_loss += total_loss.item()
       
       # Validation phase
       model.eval()
       val_loss = 0.0
       correct_action = 0
       correct_severity = 0
       total = 0
       
       with torch.no_grad():
           for videos, action_labels, severity_labels in val_loader:
               videos = videos.to(device)
               action_labels = action_labels.to(device)
               severity_labels = severity_labels.to(device)
               
               action_outputs, severity_outputs = model(videos)
               
               # Calculate validation loss
               loss_action = criterion_action(action_outputs, action_labels)
               loss_severity = criterion_severity(severity_outputs, severity_labels)
               val_loss += (loss_action + loss_severity).item()
               
               # Calculate accuracy
               _, action_pred = torch.max(action_outputs, 1)
               _, severity_pred = torch.max(severity_outputs, 1)
               
               total += action_labels.size(0)
               correct_action += (action_pred == action_labels).sum().item()
               correct_severity += (severity_pred == severity_labels).sum().item()
       
       # Print epoch results
       avg_train_loss = train_loss / len(train_loader)
       avg_val_loss = val_loss / len(val_loader)
       action_acc = 100 * correct_action / total
       severity_acc = 100 * correct_severity / total
       
       print(f'Epoch [{epoch+1}/{num_epochs}]')
       print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
       print(f'Action Acc: {action_acc:.2f}%, Severity Acc: {severity_acc:.2f}%')

Advanced Training with Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import wandb
   from tqdm import tqdm
   
   # Initialize Weights & Biases logging
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
   
   def train_with_logging(model, train_loader, val_loader, num_epochs=50):
       optimizer = optim.Adam(model.parameters(), lr=1e-4)
       scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
       
       best_val_acc = 0.0
       
       for epoch in range(num_epochs):
           # Training phase
           model.train()
           train_loss = 0.0
           train_action_correct = 0
           train_severity_correct = 0
           train_total = 0
           
           train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
           for videos, action_labels, severity_labels in train_bar:
               videos = videos.to(device)
               action_labels = action_labels.to(device)
               severity_labels = severity_labels.to(device)
               
               optimizer.zero_grad()
               
               action_outputs, severity_outputs = model(videos)
               
               loss_action = criterion_action(action_outputs, action_labels)
               loss_severity = criterion_severity(severity_outputs, severity_labels)
               total_loss = loss_action + loss_severity
               
               total_loss.backward()
               optimizer.step()
               
               # Update metrics
               train_loss += total_loss.item()
               _, action_pred = torch.max(action_outputs, 1)
               _, severity_pred = torch.max(severity_outputs, 1)
               
               train_total += action_labels.size(0)
               train_action_correct += (action_pred == action_labels).sum().item()
               train_severity_correct += (severity_pred == severity_labels).sum().item()
               
               # Update progress bar
               train_bar.set_postfix({
                   'Loss': f'{total_loss.item():.4f}',
                   'Action Acc': f'{100 * train_action_correct / train_total:.2f}%'
               })
           
           # Validation phase
           val_metrics = validate_model(model, val_loader, device)
           
           # Update learning rate
           scheduler.step()
           
           # Log metrics
           wandb.log({
               "epoch": epoch,
               "train_loss": train_loss / len(train_loader),
               "val_loss": val_metrics['loss'],
               "train_action_acc": 100 * train_action_correct / train_total,
               "train_severity_acc": 100 * train_severity_correct / train_total,
               "val_action_acc": val_metrics['action_acc'],
               "val_severity_acc": val_metrics['severity_acc'],
               "learning_rate": optimizer.param_groups[0]['lr']
           })
           
           # Save best model
           current_acc = (val_metrics['action_acc'] + val_metrics['severity_acc']) / 2
           if current_acc > best_val_acc:
               best_val_acc = current_acc
               torch.save({
                   'epoch': epoch,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'best_accuracy': best_val_acc
               }, f'best_model_epoch_{epoch}.pth.tar')
               
               print(f'New best model saved! Accuracy: {best_val_acc:.2f}%')

Dataset Examples
---------------

Loading Custom Dataset
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   import json
   from torch.utils.data import Dataset
   import cv2
   import torch
   
   class CustomFootballDataset(Dataset):
       def __init__(self, data_dir, annotations_file, transform=None):
           self.data_dir = data_dir
           self.transform = transform
           
           # Load annotations
           with open(annotations_file, 'r') as f:
               self.annotations = json.load(f)
           
           self.samples = list(self.annotations.keys())
       
       def __len__(self):
           return len(self.samples)
       
       def __getitem__(self, idx):
           sample_id = self.samples[idx]
           annotation = self.annotations[sample_id]
           
           # Load video frames
           video_path = os.path.join(self.data_dir, f"{sample_id}.mp4")
           frames = self.load_video_frames(video_path)
           
           if self.transform:
               frames = self.transform(frames)
           
           action_label = annotation['action']
           severity_label = annotation['severity']
           
           return frames, action_label, severity_label
       
       def load_video_frames(self, video_path, fps=25.0):
           cap = cv2.VideoCapture(video_path)
           frames = []
           
           frame_count = 0
           while True:
               ret, frame = cap.read()
               if not ret:
                   break
               
               # Sample at desired FPS
               if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
                   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   frames.append(frame)
               
               frame_count += 1
           
           cap.release()
           return torch.tensor(frames).permute(0, 3, 1, 2).float()
   
   # Usage
   custom_dataset = CustomFootballDataset(
       data_dir='/path/to/videos',
       annotations_file='/path/to/annotations.json',
       transform=transform
   )

Data Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torchvision.transforms as transforms
   from torchvision.transforms import functional as F
   
   class VideoPreprocessor:
       def __init__(self, target_size=(224, 224), fps=25.0):
           self.target_size = target_size
           self.fps = fps
           
           self.transform = transforms.Compose([
               transforms.Resize(target_size),
               transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]
               )
           ])
       
       def __call__(self, video_path):
           # Load video
           frames = self.load_video(video_path)
           
           # Apply transformations
           processed_frames = []
           for frame in frames:
               processed_frame = self.transform(frame)
               processed_frames.append(processed_frame)
           
           return torch.stack(processed_frames)
       
       def load_video(self, video_path):
           # Implementation for loading video frames
           pass
   
   # Usage with data augmentation
   class AugmentedVideoTransform:
       def __init__(self, training=True):
           self.training = training
       
       def __call__(self, video):
           if self.training:
               # Random horizontal flip
               if torch.rand(1) < 0.5:
                   video = F.hflip(video)
               
               # Random rotation
               if torch.rand(1) < 0.3:
                   angle = torch.randint(-10, 11, (1,)).item()
                   video = F.rotate(video, angle)
               
               # Color jittering
               if torch.rand(1) < 0.4:
                   brightness = 0.8 + torch.rand(1) * 0.4
                   video = F.adjust_brightness(video, brightness)
           
           # Standard preprocessing
           video = F.resize(video, (224, 224))
           video = F.normalize(
               video,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225]
           )
           
           return video

Interface Examples
-----------------

GUI Application Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sys
   from PyQt5.QtWidgets import QApplication
   from interface.video_window import VideoWindow
   
   def run_vars_interface():
       # Create Qt application
       app = QApplication(sys.argv)
       
       # Create main window
       main_window = VideoWindow()
       
       # Customize window settings
       main_window.setWindowTitle("VARS - Video Assistant Referee System")
       main_window.showMaximized()
       
       # Run application
       sys.exit(app.exec_())
   
   if __name__ == '__main__':
       run_vars_interface()

Programmatic Video Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from interface.video_window import VideoWindow
   from PyQt5.QtWidgets import QApplication
   import sys
   
   class VARSAnalyzer:
       def __init__(self):
           self.app = QApplication(sys.argv)
           self.video_window = VideoWindow()
       
       def analyze_video(self, video_path):
           """Analyze a video file programmatically"""
           
           # Load video
           self.video_window.load_video_file(video_path)
           
           # Run prediction
           results = self.video_window.predict_action()
           
           return {
               'video_path': video_path,
               'action_prediction': results['action'],
               'severity_prediction': results['severity'],
               'confidence_scores': results['confidence']
           }
       
       def batch_analyze(self, video_paths):
           """Analyze multiple videos"""
           results = []
           
           for video_path in video_paths:
               try:
                   result = self.analyze_video(video_path)
                   results.append(result)
                   print(f"Analyzed: {video_path}")
               except Exception as e:
                   print(f"Error analyzing {video_path}: {e}")
           
           return results
   
   # Usage
   analyzer = VARSAnalyzer()
   video_files = [
       '/path/to/video1.mp4',
       '/path/to/video2.mp4',
       '/path/to/video3.mp4'
   ]
   
   results = analyzer.batch_analyze(video_files)
   for result in results:
       print(f"Video: {result['video_path']}")
       print(f"Action: {result['action_prediction']}")
       print(f"Severity: {result['severity_prediction']}")

Evaluation Examples
------------------

Model Comparison
~~~~~~~~~~~~~~~

.. code-block:: python

   from src.evaluation import MultiTaskEvaluator
   from sklearn.metrics import classification_report
   import matplotlib.pyplot as plt
   
   def compare_models(models, test_dataset, device):
       """Compare multiple models on the same test dataset"""
       
       test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
       results = {}
       
       for model_name, model in models.items():
           print(f"Evaluating {model_name}...")
           
           evaluator = MultiTaskEvaluator(
               action_classes=test_dataset.action_classes,
               severity_classes=test_dataset.severity_classes
           )
           
           metrics = evaluator.evaluate(model, test_loader, device)
           results[model_name] = metrics
           
           print(f"{model_name} Results:")
           print(f"  Action Accuracy: {metrics['action_accuracy']:.2f}%")
           print(f"  Severity Accuracy: {metrics['severity_accuracy']:.2f}%")
       
       return results
   
   # Usage
   models = {
       'MViT + Attention': MVNetwork('mvit_v2_s', 'attention'),
       'MViT + Max Pool': MVNetwork('mvit_v2_s', 'max'),
       'R3D + Attention': MVNetwork('r3d_18', 'attention')
   }
   
   # Load pre-trained weights for each model
   for model_name, model in models.items():
       checkpoint_path = f'models/{model_name.lower().replace(" ", "_")}.pth.tar'
       checkpoint = torch.load(checkpoint_path, map_location=device)
       model.load_state_dict(checkpoint['state_dict'])
       model.eval()
   
   comparison_results = compare_models(models, test_dataset, device)

Performance Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   import numpy as np
   
   def plot_model_comparison(results):
       """Create visualization comparing model performance"""
       
       models = list(results.keys())
       action_accs = [results[model]['action_accuracy'] for model in models]
       severity_accs = [results[model]['severity_accuracy'] for model in models]
       
       x = np.arange(len(models))
       width = 0.35
       
       fig, ax = plt.subplots(figsize=(12, 6))
       
       bars1 = ax.bar(x - width/2, action_accs, width, label='Action Accuracy', alpha=0.8)
       bars2 = ax.bar(x + width/2, severity_accs, width, label='Severity Accuracy', alpha=0.8)
       
       ax.set_xlabel('Models')
       ax.set_ylabel('Accuracy (%)')
       ax.set_title('Model Performance Comparison')
       ax.set_xticks(x)
       ax.set_xticklabels(models, rotation=45, ha='right')
       ax.legend()
       
       # Add value labels on bars
       for bar in bars1:
           height = bar.get_height()
           ax.annotate(f'{height:.1f}%',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
       
       for bar in bars2:
           height = bar.get_height()
           ax.annotate(f'{height:.1f}%',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
       
       plt.tight_layout()
       plt.show()
   
   # Create comparison plot
   plot_model_comparison(comparison_results)

Error Analysis
~~~~~~~~~~~~~

.. code-block:: python

   def analyze_errors(model, test_loader, device, class_names):
       """Analyze model errors to identify improvement areas"""
       
       model.eval()
       errors = []
       
       with torch.no_grad():
           for batch_idx, (videos, action_labels, severity_labels) in enumerate(test_loader):
               videos = videos.to(device)
               action_outputs, severity_outputs = model(videos)
               
               action_preds = torch.argmax(action_outputs, dim=1)
               severity_preds = torch.argmax(severity_outputs, dim=1)
               
               # Find errors
               action_errors = action_preds != action_labels.to(device)
               severity_errors = severity_preds != severity_labels.to(device)
               
               for i in range(len(videos)):
                   if action_errors[i] or severity_errors[i]:
                       # Get confidence scores
                       action_probs = torch.softmax(action_outputs[i], dim=0)
                       severity_probs = torch.softmax(severity_outputs[i], dim=0)
                       
                       errors.append({
                           'batch_idx': batch_idx,
                           'sample_idx': i,
                           'action_true': action_labels[i].item(),
                           'action_pred': action_preds[i].item(),
                           'action_confidence': action_probs.max().item(),
                           'severity_true': severity_labels[i].item(),
                           'severity_pred': severity_preds[i].item(),
                           'severity_confidence': severity_probs.max().item(),
                           'action_error': action_errors[i].item(),
                           'severity_error': severity_errors[i].item()
                       })
       
       # Analyze error patterns
       action_error_analysis = {}
       for error in errors:
           if error['action_error']:
               true_class = class_names['action'][error['action_true']]
               pred_class = class_names['action'][error['action_pred']]
               key = f"{true_class} -> {pred_class}"
               
               if key not in action_error_analysis:
                   action_error_analysis[key] = []
               action_error_analysis[key].append(error['action_confidence'])
       
       # Print most common errors
       print("Most Common Action Classification Errors:")
       for error_type, confidences in sorted(action_error_analysis.items(), 
                                           key=lambda x: len(x[1]), reverse=True)[:5]:
           avg_confidence = np.mean(confidences)
           print(f"  {error_type}: {len(confidences)} errors, avg confidence: {avg_confidence:.3f}")
       
       return errors

Real-time Processing
-------------------

Live Video Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import threading
   import queue
   from collections import deque
   
   class RealTimeVARS:
       def __init__(self, model, device, buffer_size=50):
           self.model = model
           self.device = device
           self.buffer_size = buffer_size
           
           self.frame_buffer = deque(maxlen=buffer_size)
           self.prediction_queue = queue.Queue()
           self.running = False
       
       def start_processing(self, video_source=0):
           """Start real-time video processing"""
           
           self.running = True
           
           # Start video capture
           cap = cv2.VideoCapture(video_source)
           cap.set(cv2.CAP_PROP_FPS, 25)
           
           # Start prediction thread
           prediction_thread = threading.Thread(target=self._prediction_worker)
           prediction_thread.start()
           
           try:
               while self.running:
                   ret, frame = cap.read()
                   if not ret:
                       break
                   
                   # Add frame to buffer
                   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   self.frame_buffer.append(frame_rgb)
                   
                   # Display frame with predictions
                   self._display_frame_with_predictions(frame)
                   
                   if cv2.waitKey(1) & 0xFF == ord('q'):
                       break
           
           finally:
               self.running = False
               cap.release()
               cv2.destroyAllWindows()
               prediction_thread.join()
       
       def _prediction_worker(self):
           """Worker thread for running predictions"""
           
           while self.running:
               if len(self.frame_buffer) >= 25:  # 1 second of frames
                   # Extract frames for prediction
                   frames = list(self.frame_buffer)[-25:]  # Last 25 frames
                   
                   # Preprocess frames
                   processed_frames = self._preprocess_frames(frames)
                   
                   # Run prediction
                   with torch.no_grad():
                       action_output, severity_output = self.model(processed_frames)
                       
                       action_pred = torch.argmax(action_output, dim=1).item()
                       severity_pred = torch.argmax(severity_output, dim=1).item()
                       
                       # Store prediction
                       prediction = {
                           'action': action_pred,
                           'severity': severity_pred,
                           'timestamp': time.time()
                       }
                       
                       if not self.prediction_queue.full():
                           self.prediction_queue.put(prediction)
               
               time.sleep(0.1)  # Small delay
       
       def _preprocess_frames(self, frames):
           """Preprocess frames for model input"""
           # Convert to tensor and apply transformations
           # Implementation details...
           pass
       
       def _display_frame_with_predictions(self, frame):
           """Display frame with prediction overlay"""
           
           # Get latest prediction
           if not self.prediction_queue.empty():
               prediction = self.prediction_queue.get()
               
               # Add prediction text to frame
               action_text = f"Action: {prediction['action']}"
               severity_text = f"Severity: {prediction['severity']}"
               
               cv2.putText(frame, action_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               cv2.putText(frame, severity_text, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
           cv2.imshow('VARS Real-time Analysis', frame)
   
   # Usage
   model = MVNetwork('mvit_v2_s', 'attention')
   # Load model weights...
   
   realtime_vars = RealTimeVARS(model, device)
   realtime_vars.start_processing(video_source='path/to/video.mp4')

Utility Examples
---------------

Video Processing Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np
   from moviepy.editor import VideoFileClip
   
   def extract_action_clips(video_path, annotations, output_dir, clip_duration=4):
       """Extract action clips from full match videos"""
       
       video = VideoFileClip(video_path)
       
       for i, annotation in enumerate(annotations):
           start_time = annotation['timestamp'] - clip_duration // 2
           end_time = annotation['timestamp'] + clip_duration // 2
           
           # Ensure valid time bounds
           start_time = max(0, start_time)
           end_time = min(video.duration, end_time)
           
           # Extract clip
           clip = video.subclip(start_time, end_time)
           
           # Save clip
           output_path = os.path.join(output_dir, f"action_{i:04d}.mp4")
           clip.write_videofile(output_path, codec='libx264')
           
           print(f"Extracted clip {i+1}/{len(annotations)}: {output_path}")
       
       video.close()
   
   def resize_videos(input_dir, output_dir, target_size=(224, 224)):
       """Resize all videos in a directory"""
       
       os.makedirs(output_dir, exist_ok=True)
       
       for filename in os.listdir(input_dir):
           if filename.endswith(('.mp4', '.avi', '.mov')):
               input_path = os.path.join(input_dir, filename)
               output_path = os.path.join(output_dir, filename)
               
               cap = cv2.VideoCapture(input_path)
               fps = cap.get(cv2.CAP_PROP_FPS)
               
               fourcc = cv2.VideoWriter_fourcc(*'mp4v')
               out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
               
               while True:
                   ret, frame = cap.read()
                   if not ret:
                       break
                   
                   resized_frame = cv2.resize(frame, target_size)
                   out.write(resized_frame)
               
               cap.release()
               out.release()
               
               print(f"Resized: {filename}")

Data Analysis Utilities
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   
   def analyze_dataset_distribution(annotations_file):
       """Analyze the distribution of classes in the dataset"""
       
       with open(annotations_file, 'r') as f:
           annotations = json.load(f)
       
       # Extract labels
       action_labels = [ann['action'] for ann in annotations.values()]
       severity_labels = [ann['severity'] for ann in annotations.values()]
       
       # Create distribution plots
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
       
       # Action distribution
       action_counts = pd.Series(action_labels).value_counts()
       action_counts.plot(kind='bar', ax=ax1, title='Action Class Distribution')
       ax1.set_xlabel('Action Class')
       ax1.set_ylabel('Count')
       ax1.tick_params(axis='x', rotation=45)
       
       # Severity distribution
       severity_counts = pd.Series(severity_labels).value_counts()
       severity_counts.plot(kind='bar', ax=ax2, title='Severity Class Distribution')
       ax2.set_xlabel('Severity Class')
       ax2.set_ylabel('Count')
       ax2.tick_params(axis='x', rotation=45)
       
       plt.tight_layout()
       plt.show()
       
       return {
           'action_distribution': action_counts.to_dict(),
           'severity_distribution': severity_counts.to_dict(),
           'total_samples': len(annotations)
       }
   
   def export_predictions_to_csv(predictions, output_file):
       """Export model predictions to CSV format"""
       
       df = pd.DataFrame(predictions)
       df.to_csv(output_file, index=False)
       
       print(f"Predictions exported to {output_file}")
       print(f"Total samples: {len(df)}")
       print(f"Columns: {list(df.columns)}")

Configuration Examples
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # config.py - Centralized configuration
   
   class VARSConfig:
       # Model configuration
       MODEL_NAME = "mvit_v2_s"
       AGGREGATION_TYPE = "attention"
       NUM_CLASSES_ACTION = 8
       NUM_CLASSES_SEVERITY = 4
       
       # Training configuration
       BATCH_SIZE = 8
       LEARNING_RATE = 1e-4
       NUM_EPOCHS = 50
       WEIGHT_DECAY = 1e-4
       
       # Data configuration
       FPS = 25.0
       NUM_VIEWS = 5
       INPUT_SIZE = (224, 224)
       CLIP_DURATION = 2.0  # seconds
       
       # Dataset paths
       DATASET_PATH = "/path/to/soccernet-mvfoul"
       PRETRAINED_WEIGHTS = "/path/to/pretrained/model.pth.tar"
       
       # Class names
       ACTION_CLASSES = [
           'No action', 'Throwing', 'Other', 'Tackling',
           'Standing tackling', 'High leg', 'Holding', 'Pushing'
       ]
       
       SEVERITY_CLASSES = [
           'No offence', 'No card', 'Yellow card', 'Red card'
       ]
   
   # Usage in training script
   from config import VARSConfig
   
   def main():
       config = VARSConfig()
       
       model = MVNetwork(
           net_name=config.MODEL_NAME,
           agr_type=config.AGGREGATION_TYPE,
           num_classes_action=config.NUM_CLASSES_ACTION,
           num_classes_severity=config.NUM_CLASSES_SEVERITY
       )
       
       # Use other configuration parameters...

These examples provide practical starting points for using the VARS system in various scenarios, from basic inference to advanced training and real-time processing applications.
