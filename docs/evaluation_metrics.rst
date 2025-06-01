Evaluation Metrics
==================

This section provides comprehensive documentation on the evaluation metrics used in the VARS system for assessing model performance on football foul detection tasks.

Overview
--------

The VARS system uses multi-task evaluation, assessing performance on both action classification and offence severity prediction. The evaluation framework provides detailed metrics for each task and overall system performance.

**Key Evaluation Areas:**

- **Action Classification**: 8-class classification of football actions
- **Severity Assessment**: 4-class offence severity prediction  
- **Multi-View Analysis**: Performance across different camera angles
- **Temporal Consistency**: Stability of predictions over time

Core Metrics
------------

Accuracy Metrics
~~~~~~~~~~~~~~~~

**Overall Accuracy**: Percentage of correctly classified samples

.. math::
   
   \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}

.. code-block:: python

   def calculate_accuracy(predictions, labels):
       correct = (predictions == labels).sum().item()
       total = labels.size(0)
       return 100.0 * correct / total

**Top-k Accuracy**: Percentage of samples where the true label is in the top-k predictions

.. code-block:: python

   def top_k_accuracy(output, target, topk=(1, 5)):
       maxk = max(topk)
       batch_size = target.size(0)
       
       _, pred = output.topk(maxk, 1, True, True)
       pred = pred.t()
       correct = pred.eq(target.view(1, -1).expand_as(pred))
       
       res = []
       for k in topk:
           correct_k = correct[:k].view(-1).float().sum(0)
           res.append(correct_k.mul_(100.0 / batch_size))
       return res

Precision, Recall, and F1-Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Per-Class Metrics**: Detailed performance for each action class

.. code-block:: python

   from sklearn.metrics import precision_recall_fscore_support, classification_report
   
   def detailed_classification_metrics(y_true, y_pred, class_names):
       # Calculate precision, recall, F1 for each class
       precision, recall, f1, support = precision_recall_fscore_support(
           y_true, y_pred, average=None
       )
       
       # Create detailed report
       report = classification_report(
           y_true, y_pred, 
           target_names=class_names,
           output_dict=True
       )
       
       return {
           'precision': precision,
           'recall': recall,
           'f1_score': f1,
           'support': support,
           'detailed_report': report
       }

**Macro and Micro Averages**:

.. code-block:: python

   # Macro average (unweighted mean of class metrics)
   macro_precision = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
   macro_recall = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
   macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
   
   # Micro average (global calculation)
   micro_precision = precision_recall_fscore_support(y_true, y_pred, average='micro')[0]
   micro_recall = precision_recall_fscore_support(y_true, y_pred, average='micro')[1]
   micro_f1 = precision_recall_fscore_support(y_true, y_pred, average='micro')[2]

Confusion Matrix Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.metrics import confusion_matrix
   
   def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
       cm = confusion_matrix(y_true, y_pred)
       
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
       plt.ylabel('True Label')
       plt.xlabel('Predicted Label')
       plt.tight_layout()
       plt.show()
       
       return cm

Multi-Task Evaluation
---------------------

Joint Performance Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskEvaluator:
       def __init__(self, action_classes, severity_classes):
           self.action_classes = action_classes
           self.severity_classes = severity_classes
       
       def evaluate(self, model, test_loader, device):
           model.eval()
           
           # Storage for predictions and labels
           action_preds, action_labels = [], []
           severity_preds, severity_labels = [], []
           
           with torch.no_grad():
               for videos, actions, severities in test_loader:
                   videos = videos.to(device)
                   
                   # Get predictions
                   action_out, severity_out = model(videos)
                   
                   # Convert to class predictions
                   _, action_pred = torch.max(action_out, 1)
                   _, severity_pred = torch.max(severity_out, 1)
                   
                   # Store results
                   action_preds.extend(action_pred.cpu().numpy())
                   severity_preds.extend(severity_pred.cpu().numpy())
                   action_labels.extend(actions.numpy())
                   severity_labels.extend(severities.numpy())
           
           # Calculate metrics for both tasks
           action_metrics = detailed_classification_metrics(
               action_labels, action_preds, self.action_classes
           )
           severity_metrics = detailed_classification_metrics(
               severity_labels, severity_preds, self.severity_classes
           )
           
           return {
               'action_metrics': action_metrics,
               'severity_metrics': severity_metrics,
               'action_accuracy': calculate_accuracy(
                   torch.tensor(action_preds), torch.tensor(action_labels)
               ),
               'severity_accuracy': calculate_accuracy(
                   torch.tensor(severity_preds), torch.tensor(severity_labels)
               )
           }

Task Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_task_correlation(action_preds, severity_preds, action_labels, severity_labels):
       """Analyze correlation between action and severity predictions"""
       
       # Create contingency tables
       action_severity_pred = pd.crosstab(
           pd.Series(action_preds, name='Action'), 
           pd.Series(severity_preds, name='Severity')
       )
       
       action_severity_true = pd.crosstab(
           pd.Series(action_labels, name='Action'), 
           pd.Series(severity_labels, name='Severity')
       )
       
       # Calculate correlation coefficients
       from scipy.stats import pearsonr, spearmanr
       
       # Correlation between predicted action and severity
       corr_pred_pearson, p_pred_pearson = pearsonr(action_preds, severity_preds)
       corr_pred_spearman, p_pred_spearman = spearmanr(action_preds, severity_preds)
       
       # Correlation between true action and severity
       corr_true_pearson, p_true_pearson = pearsonr(action_labels, severity_labels)
       corr_true_spearman, p_true_spearman = spearmanr(action_labels, severity_labels)
       
       return {
           'pred_correlations': {
               'pearson': (corr_pred_pearson, p_pred_pearson),
               'spearman': (corr_pred_spearman, p_pred_spearman)
           },
           'true_correlations': {
               'pearson': (corr_true_pearson, p_true_pearson),
               'spearman': (corr_true_spearman, p_true_spearman)
           },
           'contingency_tables': {
               'predictions': action_severity_pred,
               'ground_truth': action_severity_true
           }
       }

Performance Visualization
-------------------------

ROC Curves and AUC
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.metrics import roc_curve, auc
   from sklearn.preprocessing import label_binarize
   import matplotlib.pyplot as plt
   
   def plot_multiclass_roc(y_true, y_score, class_names, title='ROC Curves'):
       """Plot ROC curves for multi-class classification"""
       
       # Binarize the output
       y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
       n_classes = y_true_bin.shape[1]
       
       # Compute ROC curve and ROC area for each class
       fpr = dict()
       tpr = dict()
       roc_auc = dict()
       
       for i in range(n_classes):
           fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
           roc_auc[i] = auc(fpr[i], tpr[i])
       
       # Plot ROC curves
       plt.figure(figsize=(12, 8))
       colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
       
       for i, color in zip(range(n_classes), colors):
           plt.plot(
               fpr[i], tpr[i], color=color, lw=2,
               label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
           )
       
       plt.plot([0, 1], [0, 1], 'k--', lw=2)
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title(title)
       plt.legend(loc="lower right")
       plt.grid(True)
       plt.show()
       
       return roc_auc

Precision-Recall Curves
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.metrics import precision_recall_curve, average_precision_score
   
   def plot_precision_recall_curves(y_true, y_score, class_names):
       """Plot precision-recall curves for multi-class classification"""
       
       y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
       n_classes = y_true_bin.shape[1]
       
       plt.figure(figsize=(12, 8))
       colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
       
       for i, color in zip(range(n_classes), colors):
           precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
           avg_precision = average_precision_score(y_true_bin[:, i], y_score[:, i])
           
           plt.plot(
               recall, precision, color=color, lw=2,
               label=f'{class_names[i]} (AP = {avg_precision:.2f})'
           )
       
       plt.xlabel('Recall')
       plt.ylabel('Precision')
       plt.title('Precision-Recall Curves')
       plt.legend(loc="lower left")
       plt.grid(True)
       plt.show()

Performance Heatmaps
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_performance_heatmap(metrics_dict, title='Performance Heatmap'):
       """Create heatmap showing performance across different metrics and classes"""
       
       # Extract metrics for visualization
       classes = list(metrics_dict.keys())
       metric_names = ['Precision', 'Recall', 'F1-Score']
       
       data = []
       for class_name in classes:
           class_metrics = metrics_dict[class_name]
           data.append([
               class_metrics['precision'],
               class_metrics['recall'], 
               class_metrics['f1-score']
           ])
       
       data = np.array(data)
       
       plt.figure(figsize=(10, 8))
       sns.heatmap(
           data,
           annot=True,
           fmt='.3f',
           cmap='RdYlBu_r',
           xticklabels=metric_names,
           yticklabels=classes,
           cbar_kws={'label': 'Score'}
       )
       plt.title(title)
       plt.tight_layout()
       plt.show()

Temporal Analysis
-----------------

Prediction Stability
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_temporal_stability(model, video_sequences, device, window_size=5):
       """Analyze prediction stability over time windows"""
       
       model.eval()
       stability_scores = []
       
       with torch.no_grad():
           for sequence in video_sequences:
               predictions = []
               
               # Get predictions for overlapping windows
               for start_idx in range(0, len(sequence) - window_size + 1):
                   window = sequence[start_idx:start_idx + window_size]
                   window_tensor = torch.stack(window).unsqueeze(0).to(device)
                   
                   action_out, severity_out = model(window_tensor)
                   action_pred = torch.argmax(action_out, dim=1).cpu().item()
                   severity_pred = torch.argmax(severity_out, dim=1).cpu().item()
                   
                   predictions.append((action_pred, severity_pred))
               
               # Calculate stability (consistency of predictions)
               if len(predictions) > 1:
                   action_changes = sum(
                       1 for i in range(1, len(predictions)) 
                       if predictions[i][0] != predictions[i-1][0]
                   )
                   severity_changes = sum(
                       1 for i in range(1, len(predictions)) 
                       if predictions[i][1] != predictions[i-1][1]
                   )
                   
                   action_stability = 1.0 - (action_changes / (len(predictions) - 1))
                   severity_stability = 1.0 - (severity_changes / (len(predictions) - 1))
                   
                   stability_scores.append({
                       'action_stability': action_stability,
                       'severity_stability': severity_stability
                   })
       
       return stability_scores

Multi-View Performance
----------------------

View-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def evaluate_per_view_performance(model, multi_view_loader, device, num_views=5):
       """Evaluate performance for each camera view separately"""
       
       model.eval()
       view_performances = {i: {'correct': 0, 'total': 0} for i in range(num_views)}
       
       with torch.no_grad():
           for batch_idx, (videos, labels) in enumerate(multi_view_loader):
               # videos shape: (batch, views, channels, frames, height, width)
               batch_size = videos.size(0)
               
               for view_idx in range(num_views):
                   # Extract single view
                   single_view = videos[:, view_idx].to(device)
                   
                   # Get predictions for this view
                   outputs = model.encoder(single_view)  # Use encoder only
                   predictions = torch.argmax(outputs, dim=1)
                   
                   # Calculate accuracy for this view
                   correct = (predictions == labels.to(device)).sum().item()
                   view_performances[view_idx]['correct'] += correct
                   view_performances[view_idx]['total'] += batch_size
       
       # Calculate accuracy per view
       view_accuracies = {}
       for view_idx in range(num_views):
           accuracy = 100.0 * view_performances[view_idx]['correct'] / view_performances[view_idx]['total']
           view_accuracies[f'view_{view_idx}'] = accuracy
       
       return view_accuracies

Aggregation Method Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_aggregation_methods(features, labels, methods=['max', 'mean', 'attention']):
       """Compare different view aggregation methods"""
       
       results = {}
       
       for method in methods:
           if method == 'max':
               aggregated = torch.max(features, dim=1)[0]
           elif method == 'mean':
               aggregated = torch.mean(features, dim=1)
           elif method == 'attention':
               # Simplified attention mechanism
               attention_weights = torch.softmax(
                   torch.mean(features, dim=-1), dim=1
               ).unsqueeze(-1)
               aggregated = torch.sum(features * attention_weights, dim=1)
           
           # Calculate accuracy with this aggregation method
           # (This would typically involve a classifier head)
           # For demonstration, we'll use a simple distance-based approach
           
           results[method] = {
               'aggregated_features': aggregated,
               'method_name': method
           }
       
       return results

Statistical Analysis
--------------------

Significance Testing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy.stats import ttest_rel, wilcoxon
   import numpy as np
   
   def statistical_comparison(results_a, results_b, metric='accuracy'):
       """Compare two model results statistically"""
       
       scores_a = [r[metric] for r in results_a]
       scores_b = [r[metric] for r in results_b]
       
       # Paired t-test
       t_stat, t_p_value = ttest_rel(scores_a, scores_b)
       
       # Wilcoxon signed-rank test (non-parametric)
       w_stat, w_p_value = wilcoxon(scores_a, scores_b)
       
       # Effect size (Cohen's d)
       pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
       cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
       
       return {
           'paired_ttest': {'statistic': t_stat, 'p_value': t_p_value},
           'wilcoxon': {'statistic': w_stat, 'p_value': w_p_value},
           'effect_size': cohens_d,
           'mean_difference': np.mean(scores_a) - np.mean(scores_b)
       }

Cross-Validation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold
   
   def cross_validation_evaluation(model_class, dataset, k_folds=5):
       """Perform k-fold cross-validation evaluation"""
       
       skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
       fold_results = []
       
       for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.samples, dataset.labels)):
           print(f"Evaluating fold {fold + 1}/{k_folds}")
           
           # Create train/validation splits
           train_subset = torch.utils.data.Subset(dataset, train_idx)
           val_subset = torch.utils.data.Subset(dataset, val_idx)
           
           train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
           val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
           
           # Initialize and train model
           model = model_class()
           # ... training code ...
           
           # Evaluate on validation set
           evaluator = MultiTaskEvaluator(action_classes, severity_classes)
           fold_metrics = evaluator.evaluate(model, val_loader, device)
           
           fold_results.append({
               'fold': fold,
               'action_accuracy': fold_metrics['action_accuracy'],
               'severity_accuracy': fold_metrics['severity_accuracy'],
               'detailed_metrics': fold_metrics
           })
       
       # Calculate cross-validation statistics
       action_accuracies = [r['action_accuracy'] for r in fold_results]
       severity_accuracies = [r['severity_accuracy'] for r in fold_results]
       
       cv_results = {
           'action_accuracy': {
               'mean': np.mean(action_accuracies),
               'std': np.std(action_accuracies),
               'min': np.min(action_accuracies),
               'max': np.max(action_accuracies)
           },
           'severity_accuracy': {
               'mean': np.mean(severity_accuracies),
               'std': np.std(severity_accuracies),
               'min': np.min(severity_accuracies),
               'max': np.max(severity_accuracies)
           },
           'fold_details': fold_results
       }
       
       return cv_results

Benchmarking Framework
----------------------

Comprehensive Evaluation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class VARSBenchmark:
       def __init__(self, models, test_datasets, device):
           self.models = models
           self.test_datasets = test_datasets
           self.device = device
           self.results = {}
       
       def run_benchmark(self):
           """Run comprehensive benchmark across all models and datasets"""
           
           for model_name, model in self.models.items():
               print(f"Benchmarking {model_name}")
               self.results[model_name] = {}
               
               for dataset_name, dataset in self.test_datasets.items():
                   print(f"  Evaluating on {dataset_name}")
                   
                   # Create data loader
                   test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
                   
                   # Run evaluation
                   evaluator = MultiTaskEvaluator(
                       dataset.action_classes, 
                       dataset.severity_classes
                   )
                   
                   metrics = evaluator.evaluate(model, test_loader, self.device)
                   
                   # Add timing information
                   start_time = time.time()
                   _ = evaluator.evaluate(model, test_loader, self.device)
                   inference_time = time.time() - start_time
                   
                   self.results[model_name][dataset_name] = {
                       **metrics,
                       'inference_time': inference_time,
                       'fps': len(dataset) / inference_time
                   }
       
       def generate_report(self):
           """Generate comprehensive benchmark report"""
           
           report = "# VARS Model Benchmark Report\n\n"
           
           # Summary table
           report += "## Summary\n\n"
           report += "| Model | Dataset | Action Acc | Severity Acc | FPS |\n"
           report += "|-------|---------|------------|--------------|-----|\n"
           
           for model_name in self.results:
               for dataset_name in self.results[model_name]:
                   result = self.results[model_name][dataset_name]
                   report += f"| {model_name} | {dataset_name} | "
                   report += f"{result['action_accuracy']:.2f}% | "
                   report += f"{result['severity_accuracy']:.2f}% | "
                   report += f"{result['fps']:.1f} |\n"
           
           # Detailed analysis
           report += "\n## Detailed Analysis\n\n"
           
           for model_name in self.results:
               report += f"### {model_name}\n\n"
               
               for dataset_name in self.results[model_name]:
                   result = self.results[model_name][dataset_name]
                   report += f"**{dataset_name}:**\n"
                   report += f"- Action Accuracy: {result['action_accuracy']:.2f}%\n"
                   report += f"- Severity Accuracy: {result['severity_accuracy']:.2f}%\n"
                   report += f"- Inference Speed: {result['fps']:.1f} FPS\n"
                   report += f"- Total Inference Time: {result['inference_time']:.2f}s\n\n"
           
           return report

Export and Reporting
--------------------

Results Export
~~~~~~~~~~~~~~

.. code-block:: python

   import json
   import csv
   from datetime import datetime
   
   def export_evaluation_results(results, format='json', filename=None):
       """Export evaluation results in various formats"""
       
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       
       if filename is None:
           filename = f"vars_evaluation_{timestamp}"
       
       if format == 'json':
           with open(f"{filename}.json", 'w') as f:
               json.dump(results, f, indent=2)
       
       elif format == 'csv':
           # Flatten results for CSV export
           flattened = []
           for model_name, model_results in results.items():
               for dataset_name, metrics in model_results.items():
                   row = {
                       'model': model_name,
                       'dataset': dataset_name,
                       **metrics
                   }
                   flattened.append(row)
           
           with open(f"{filename}.csv", 'w', newline='') as f:
               if flattened:
                   writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
                   writer.writeheader()
                   writer.writerows(flattened)
       
       print(f"Results exported to {filename}.{format}")

Report Generation
~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_evaluation_report(results, output_path='evaluation_report.md'):
       """Generate a comprehensive evaluation report in Markdown format"""
       
       report_content = f"""# VARS Evaluation Report
   
   Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
   
   ## Executive Summary
   
   This report presents the evaluation results of the VARS (Video Assistant Referee System) 
   for football foul detection and severity assessment.
   
   ## Model Performance Overview
   
   """
       
       # Add performance tables and visualizations
       # ... (report generation code) ...
       
       with open(output_path, 'w') as f:
           f.write(report_content)
       
       print(f"Evaluation report generated: {output_path}")

Usage Examples
--------------

Complete Evaluation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize models and datasets
   models = {
       'mvit_attention': MVNetwork('mvit_v2_s', 'attention'),
       'mvit_max': MVNetwork('mvit_v2_s', 'max'),
       'r3d_attention': MVNetwork('r3d_18', 'attention')
   }
   
   test_datasets = {
       'soccernet_test': test_dataset,
       'custom_validation': custom_dataset
   }
   
   # Run comprehensive benchmark
   benchmark = VARSBenchmark(models, test_datasets, device)
   benchmark.run_benchmark()
   
   # Generate and export results
   report = benchmark.generate_report()
   print(report)
   
   # Export detailed results
   export_evaluation_results(benchmark.results, format='json')
   export_evaluation_results(benchmark.results, format='csv')

This comprehensive evaluation framework provides thorough assessment capabilities for the VARS system, enabling detailed performance analysis and comparison across different model configurations and datasets.
