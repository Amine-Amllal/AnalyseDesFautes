Evaluation API Reference
========================

This section provides detailed API documentation for the VARS evaluation components.

Evaluation Functions
-------------------

The evaluation module provides comprehensive metrics for assessing model performance on multi-view foul recognition tasks.

SoccerNet Evaluation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
   
   def evaluate_model(predictions_file, ground_truth_file):
       """
       Evaluate model predictions using SoccerNet official metrics.
       
       Args:
           predictions_file (str): Path to predictions JSON file
           ground_truth_file (str): Path to ground truth annotations
           
       Returns:
           dict: Evaluation results with balanced accuracies
       """

**Evaluation Metrics:**

The evaluation computes balanced accuracy for two main tasks:

1. **Action Classification** (8 classes)
2. **Offence & Severity Classification** (4 classes)

The final leaderboard score is the mean of both balanced accuracies.

Custom Evaluation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**evaluate_predictions()**

.. code-block:: python

   def evaluate_predictions(model, data_loader, device='cuda'):
       """
       Evaluate model on a dataset and compute metrics.
       
       Args:
           model (nn.Module): Trained VARS model
           data_loader (DataLoader): Data loader for evaluation
           device (str): Device for computation
           
       Returns:
           dict: Dictionary containing:
               - action_accuracy: Balanced accuracy for action classification
               - offence_accuracy: Balanced accuracy for offence classification  
               - action_confusion_matrix: Confusion matrix for actions
               - offence_confusion_matrix: Confusion matrix for offences
               - per_class_metrics: Precision, recall, F1 per class
       """
       
       model.eval()
       all_action_preds = []
       all_action_labels = []
       all_offence_preds = []
       all_offence_labels = []
       
       with torch.no_grad():
           for offence_labels, action_labels, videos, _ in data_loader:
               videos = videos.to(device)
               
               # Forward pass
               offence_logits, action_logits, _ = model(videos)
               
               # Get predictions
               action_pred = torch.argmax(action_logits, dim=1)
               offence_pred = torch.argmax(offence_logits, dim=1)
               
               # Convert one-hot to class indices
               action_true = torch.argmax(action_labels, dim=1)
               offence_true = torch.argmax(offence_labels, dim=1)
               
               # Store predictions
               all_action_preds.extend(action_pred.cpu().numpy())
               all_action_labels.extend(action_true.cpu().numpy())
               all_offence_preds.extend(offence_pred.cpu().numpy())
               all_offence_labels.extend(offence_true.cpu().numpy())
       
       # Compute metrics
       results = compute_metrics(
           all_action_preds, all_action_labels,
           all_offence_preds, all_offence_labels
       )
       
       return results

**compute_metrics()**

.. code-block:: python

   from sklearn.metrics import balanced_accuracy_score, confusion_matrix
   from sklearn.metrics import classification_report, precision_recall_fscore_support
   
   def compute_metrics(action_preds, action_labels, offence_preds, offence_labels):
       """
       Compute comprehensive evaluation metrics.
       
       Args:
           action_preds (list): Predicted action classes
           action_labels (list): True action classes
           offence_preds (list): Predicted offence classes  
           offence_labels (list): True offence classes
           
       Returns:
           dict: Comprehensive metrics dictionary
       """
       
       # Balanced accuracies
       action_acc = balanced_accuracy_score(action_labels, action_preds)
       offence_acc = balanced_accuracy_score(offence_labels, offence_preds)
       
       # Confusion matrices
       action_cm = confusion_matrix(action_labels, action_preds)
       offence_cm = confusion_matrix(offence_labels, offence_preds)
       
       # Per-class metrics
       action_metrics = precision_recall_fscore_support(
           action_labels, action_preds, average=None
       )
       offence_metrics = precision_recall_fscore_support(
           offence_labels, offence_preds, average=None
       )
       
       return {
           'action_accuracy': action_acc,
           'offence_accuracy': offence_acc,
           'mean_accuracy': (action_acc + offence_acc) / 2,
           'action_confusion_matrix': action_cm,
           'offence_confusion_matrix': offence_cm,
           'action_precision': action_metrics[0],
           'action_recall': action_metrics[1],
           'action_f1': action_metrics[2],
           'offence_precision': offence_metrics[0],
           'offence_recall': offence_metrics[1],
           'offence_f1': offence_metrics[2]
       }

Output Format
------------

**Prediction JSON Format**

The evaluation expects predictions in a specific JSON format:

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
           },
           "2": {
               "Action class": "Dive",
               "Offence": "No offence", 
               "Severity": "0.0"
           }
       }
   }

**Class Mappings**

.. code-block:: python

   ACTION_CLASSES = {
       0: "Tackling",
       1: "Standing tackling", 
       2: "High leg",
       3: "Holding",
       4: "Pushing",
       5: "Elbowing", 
       6: "Challenge",
       7: "Dive"
   }
   
   OFFENCE_CLASSES = {
       0: "No offence",
       1: "Offence",
       2: "Offence", 
       3: "Offence"
   }
   
   SEVERITY_MAPPING = {
       0: "0.0",  # No offence
       1: "1.0",  # No card
       2: "2.0",  # Yellow card  
       3: "3.0"   # Red card
   }

**generate_predictions_json()**

.. code-block:: python

   def generate_predictions_json(model, data_loader, output_path, device='cuda'):
       """
       Generate predictions in required JSON format.
       
       Args:
           model (nn.Module): Trained model
           data_loader (DataLoader): Data loader for prediction
           output_path (str): Output file path
           device (str): Device for computation
       """
       
       model.eval()
       predictions = {"Actions": {}}
       
       with torch.no_grad():
           for i, (_, _, videos, action_ids) in enumerate(data_loader):
               videos = videos.to(device)
               
               # Forward pass
               offence_logits, action_logits, _ = model(videos)
               
               # Get predictions
               action_pred = torch.argmax(action_logits, dim=1).item()
               offence_pred = torch.argmax(offence_logits, dim=1).item()
               
               # Convert to required format
               action_class = ACTION_CLASSES[action_pred]
               offence_class = OFFENCE_CLASSES[offence_pred]
               severity = SEVERITY_MAPPING[offence_pred]
               
               # Store prediction
               action_id = action_ids[0] if isinstance(action_ids, list) else str(i)
               predictions["Actions"][action_id] = {
                   "Action class": action_class,
                   "Offence": offence_class,
                   "Severity": severity
               }
       
       # Save to file
       with open(output_path, 'w') as f:
           json.dump(predictions, f, indent=2)

Visualization Functions
----------------------

**plot_confusion_matrix()**

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
       """
       Plot confusion matrix with class names.
       
       Args:
           cm (np.array): Confusion matrix
           class_names (list): List of class names
           title (str): Plot title
       """
       
       plt.figure(figsize=(10, 8))
       sns.heatmap(
           cm, 
           annot=True,
           fmt='d', 
           cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names
       )
       plt.title(title)
       plt.xlabel('Predicted')
       plt.ylabel('Actual')
       plt.tight_layout()
       plt.show()

**plot_per_class_metrics()**

.. code-block:: python

   def plot_per_class_metrics(precision, recall, f1, class_names, title='Per-Class Metrics'):
       """
       Plot per-class precision, recall, and F1 scores.
       
       Args:
           precision (np.array): Precision scores per class
           recall (np.array): Recall scores per class  
           f1 (np.array): F1 scores per class
           class_names (list): List of class names
           title (str): Plot title
       """
       
       x = np.arange(len(class_names))
       width = 0.25
       
       fig, ax = plt.subplots(figsize=(12, 6))
       
       ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
       ax.bar(x, recall, width, label='Recall', alpha=0.8)
       ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
       
       ax.set_xlabel('Classes')
       ax.set_ylabel('Score')
       ax.set_title(title)
       ax.set_xticks(x)
       ax.set_xticklabels(class_names, rotation=45, ha='right')
       ax.legend()
       ax.grid(True, alpha=0.3)
       
       plt.tight_layout()
       plt.show()

Model Comparison
---------------

**compare_models()**

.. code-block:: python

   def compare_models(models, model_names, data_loader, device='cuda'):
       """
       Compare multiple models on the same dataset.
       
       Args:
           models (list): List of trained models
           model_names (list): List of model names
           data_loader (DataLoader): Data loader for evaluation
           device (str): Device for computation
           
       Returns:
           pd.DataFrame: Comparison results
       """
       
       import pandas as pd
       
       results = []
       
       for model, name in zip(models, model_names):
           metrics = evaluate_predictions(model, data_loader, device)
           
           results.append({
               'Model': name,
               'Action Accuracy': metrics['action_accuracy'],
               'Offence Accuracy': metrics['offence_accuracy'],
               'Mean Accuracy': metrics['mean_accuracy'],
               'Action F1 (Macro)': metrics['action_f1'].mean(),
               'Offence F1 (Macro)': metrics['offence_f1'].mean()
           })
       
       return pd.DataFrame(results).round(4)

**cross_validation_evaluation()**

.. code-block:: python

   from sklearn.model_selection import KFold
   
   def cross_validation_evaluation(model_class, dataset, k_folds=5, **model_kwargs):
       """
       Perform k-fold cross-validation evaluation.
       
       Args:
           model_class: Model class to instantiate
           dataset: Complete dataset
           k_folds (int): Number of folds
           **model_kwargs: Model initialization arguments
           
       Returns:
           dict: Cross-validation results
       """
       
       kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
       results = {
           'action_accuracies': [],
           'offence_accuracies': [],
           'mean_accuracies': []
       }
       
       for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
           print(f"Fold {fold + 1}/{k_folds}")
           
           # Create data loaders for this fold
           train_subset = torch.utils.data.Subset(dataset, train_idx)
           val_subset = torch.utils.data.Subset(dataset, val_idx)
           
           train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
           val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
           
           # Train model (simplified - implement full training loop)
           model = model_class(**model_kwargs)
           trained_model = train_model(model, train_loader, epochs=10)
           
           # Evaluate
           metrics = evaluate_predictions(trained_model, val_loader)
           
           results['action_accuracies'].append(metrics['action_accuracy'])
           results['offence_accuracies'].append(metrics['offence_accuracy'])
           results['mean_accuracies'].append(metrics['mean_accuracy'])
       
       # Compute statistics
       results['action_mean'] = np.mean(results['action_accuracies'])
       results['action_std'] = np.std(results['action_accuracies'])
       results['offence_mean'] = np.mean(results['offence_accuracies'])
       results['offence_std'] = np.std(results['offence_accuracies'])
       results['mean_mean'] = np.mean(results['mean_accuracies'])
       results['mean_std'] = np.std(results['mean_accuracies'])
       
       return results

Error Analysis
-------------

**analyze_errors()**

.. code-block:: python

   def analyze_errors(model, data_loader, output_dir='error_analysis', device='cuda'):
       """
       Perform detailed error analysis.
       
       Args:
           model (nn.Module): Trained model
           data_loader (DataLoader): Data loader for analysis
           output_dir (str): Directory to save analysis results
           device (str): Device for computation
       """
       
       import os
       os.makedirs(output_dir, exist_ok=True)
       
       model.eval()
       errors = {
           'action_errors': [],
           'offence_errors': [],
           'both_errors': []
       }
       
       with torch.no_grad():
           for i, (offence_labels, action_labels, videos, action_ids) in enumerate(data_loader):
               videos = videos.to(device)
               
               # Forward pass
               offence_logits, action_logits, attention = model(videos)
               
               # Get predictions and true labels
               action_pred = torch.argmax(action_logits, dim=1).item()
               offence_pred = torch.argmax(offence_logits, dim=1).item()
               action_true = torch.argmax(action_labels, dim=1).item()
               offence_true = torch.argmax(offence_labels, dim=1).item()
               
               # Check for errors
               action_error = action_pred != action_true
               offence_error = offence_pred != offence_true
               
               if action_error:
                   errors['action_errors'].append({
                       'action_id': action_ids[0] if isinstance(action_ids, list) else str(i),
                       'predicted': action_pred,
                       'true': action_true,
                       'confidence': torch.softmax(action_logits, dim=1).max().item(),
                       'attention_weights': attention.cpu().numpy() if attention is not None else None
                   })
               
               if offence_error:
                   errors['offence_errors'].append({
                       'action_id': action_ids[0] if isinstance(action_ids, list) else str(i),
                       'predicted': offence_pred,
                       'true': offence_true,
                       'confidence': torch.softmax(offence_logits, dim=1).max().item(),
                       'attention_weights': attention.cpu().numpy() if attention is not None else None
                   })
               
               if action_error and offence_error:
                   errors['both_errors'].append(action_ids[0] if isinstance(action_ids, list) else str(i))
       
       # Save error analysis
       import json
       with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
           json.dump(errors, f, indent=2, default=str)
       
       return errors

**generate_error_report()**

.. code-block:: python

   def generate_error_report(errors, class_names_action, class_names_offence):
       """
       Generate human-readable error report.
       
       Args:
           errors (dict): Error analysis results
           class_names_action (list): Action class names
           class_names_offence (list): Offence class names
           
       Returns:
           str: Formatted error report
       """
       
       report = "=== ERROR ANALYSIS REPORT ===\n\n"
       
       # Action errors
       report += f"Action Classification Errors: {len(errors['action_errors'])}\n"
       if errors['action_errors']:
           for error in errors['action_errors'][:5]:  # Show first 5
               pred_class = class_names_action[error['predicted']]
               true_class = class_names_action[error['true']]
               report += f"  {error['action_id']}: {true_class} → {pred_class} (conf: {error['confidence']:.3f})\n"
       
       report += f"\nOffence Classification Errors: {len(errors['offence_errors'])}\n"
       if errors['offence_errors']:
           for error in errors['offence_errors'][:5]:  # Show first 5
               pred_class = class_names_offence[error['predicted']]
               true_class = class_names_offence[error['true']]
               report += f"  {error['action_id']}: {true_class} → {pred_class} (conf: {error['confidence']:.3f})\n"
       
       report += f"\nBoth Tasks Wrong: {len(errors['both_errors'])}\n"
       
       return report

Benchmark Functions
------------------

**benchmark_model()**

.. code-block:: python

   import time
   
   def benchmark_model(model, data_loader, device='cuda', num_runs=100):
       """
       Benchmark model inference speed.
       
       Args:
           model (nn.Module): Model to benchmark
           data_loader (DataLoader): Data loader for benchmarking
           device (str): Device for computation
           num_runs (int): Number of inference runs
           
       Returns:
           dict: Benchmark results
       """
       
       model.eval()
       times = []
       
       # Warmup
       with torch.no_grad():
           for i, (_, _, videos, _) in enumerate(data_loader):
               if i >= 5:  # 5 warmup runs
                   break
               videos = videos.to(device)
               _ = model(videos)
       
       # Actual benchmark
       with torch.no_grad():
           for i, (_, _, videos, _) in enumerate(data_loader):
               if i >= num_runs:
                   break
               
               videos = videos.to(device)
               
               start_time = time.time()
               _ = model(videos)
               end_time = time.time()
               
               times.append((end_time - start_time) * 1000)  # Convert to ms
       
       return {
           'mean_time_ms': np.mean(times),
           'std_time_ms': np.std(times),
           'min_time_ms': np.min(times),
           'max_time_ms': np.max(times),
           'fps': 1000 / np.mean(times),  # Frames per second
           'total_runs': len(times)
       }

Usage Examples
-------------

**Complete Evaluation Pipeline:**

.. code-block:: python

   def run_complete_evaluation(model_path, dataset_path, output_dir):
       """Run complete evaluation pipeline."""
       
       # Load model
       model = load_model(model_path)
       
       # Load dataset
       dataset = MultiViewDataset(
           path=dataset_path,
           start=63, end=87, fps=17,
           split='Test', num_views=5
       )
       data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
       
       # Evaluate
       metrics = evaluate_predictions(model, data_loader)
       
       # Generate predictions JSON
       generate_predictions_json(
           model, data_loader,
           os.path.join(output_dir, 'predictions.json')
       )
       
       # Error analysis
       errors = analyze_errors(model, data_loader, output_dir)
       
       # Benchmark
       benchmark_results = benchmark_model(model, data_loader)
       
       # Create report
       report = f"""
       EVALUATION RESULTS
       ==================
       Action Accuracy: {metrics['action_accuracy']:.4f}
       Offence Accuracy: {metrics['offence_accuracy']:.4f}
       Mean Accuracy: {metrics['mean_accuracy']:.4f}
       
       PERFORMANCE
       ===========
       Inference Time: {benchmark_results['mean_time_ms']:.2f} ± {benchmark_results['std_time_ms']:.2f} ms
       FPS: {benchmark_results['fps']:.1f}
       
       ERRORS
       ======
       Action Errors: {len(errors['action_errors'])}
       Offence Errors: {len(errors['offence_errors'])}
       Both Wrong: {len(errors['both_errors'])}
       """
       
       with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
           f.write(report)
       
       return metrics, errors, benchmark_results

For more detailed examples and usage patterns, see the :doc:`../examples` section.
