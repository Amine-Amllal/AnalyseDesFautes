Troubleshooting
===============

This guide helps resolve common issues encountered when working with the VARS system. It covers installation problems, training issues, inference errors, and performance optimization.

Installation Issues
-------------------

Python Environment Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ModuleNotFoundError or ImportError

.. code-block:: bash

   ModuleNotFoundError: No module named 'torch'
   ImportError: cannot import name 'VideoReader' from 'torchvision.io'

**Solutions**:

1. **Verify Python version**:
   
   .. code-block:: bash
   
      python --version  # Should be 3.8 or higher
   
2. **Install PyTorch correctly**:
   
   .. code-block:: bash
   
      # For CPU only
      pip install torch torchvision torchaudio
      
      # For CUDA 11.8
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      
      # For CUDA 12.1
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. **Check PyTorch installation**:
   
   .. code-block:: python
   
      import torch
      print(torch.__version__)
      print(torch.cuda.is_available())  # Should be True for GPU support

**Issue**: CUDA version mismatch

.. code-block:: text

   RuntimeError: CUDA runtime error (35) : CUDA driver version is insufficient for CUDA runtime version

**Solutions**:

1. **Check CUDA driver version**:
   
   .. code-block:: bash
   
      nvidia-smi  # Check driver version
      nvcc --version  # Check CUDA toolkit version

2. **Install compatible PyTorch version**:
   
   .. code-block:: bash
   
      # Check compatibility at https://pytorch.org/get-started/locally/
      pip uninstall torch torchvision torchaudio
      pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~

**Issue**: Package version conflicts

.. code-block:: text

   ERROR: pip's dependency resolver does not currently support backtracking

**Solutions**:

1. **Use conda for environment management**:
   
   .. code-block:: bash
   
      conda create -n vars python=3.9
      conda activate vars
      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

2. **Install with pip-tools for dependency resolution**:
   
   .. code-block:: bash
   
      pip install pip-tools
      pip-compile requirements.in
      pip-sync requirements.txt

3. **Use virtual environment with specific versions**:
   
   .. code-block:: bash
   
      python -m venv vars_env
      source vars_env/bin/activate  # On Windows: vars_env\Scripts\activate
      pip install -r requirements_fixed.txt

Dataset Issues
--------------

SoccerNet-MVFoul Access Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Authentication or download failures

.. code-block:: text

   Permission denied: Unable to access SoccerNet dataset
   SSL: CERTIFICATE_VERIFY_FAILED

**Solutions**:

1. **Verify SoccerNet registration**:
   - Ensure you have a valid SoccerNet account
   - Check that you've agreed to the terms of use
   - Verify your academic email is confirmed

2. **Update SSL certificates**:
   
   .. code-block:: bash
   
      # On macOS
      /Applications/Python\ 3.x/Install\ Certificates.command
      
      # On Linux/Windows
      pip install --upgrade certifi

3. **Manual download workaround**:
   
   .. code-block:: python
   
      import ssl
      ssl._create_default_https_context = ssl._create_unverified_context

**Issue**: Incomplete dataset download

.. code-block:: text

   FileNotFoundError: Dataset incomplete, missing files

**Solutions**:

1. **Resume interrupted download**:
   
   .. code-block:: python
   
      from SoccerNet.Downloader import SoccerNetDownloader
      
      mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/dataset")
      mySoccerNetDownloader.downloadGames(files=["MVFoul"], split=["train", "valid", "test"])

2. **Verify dataset integrity**:
   
   .. code-block:: python
   
      def verify_dataset(dataset_path):
           expected_files = {
               'train': 1762,
               'valid': 567, 
               'test': 1572
           }
           
           for split, expected_count in expected_files.items():
               split_path = os.path.join(dataset_path, split)
               actual_count = len(os.listdir(split_path))
               
               if actual_count != expected_count:
                   print(f"Missing files in {split}: expected {expected_count}, found {actual_count}")
               else:
                   print(f"{split} split complete: {actual_count} files")

Data Format Issues
~~~~~~~~~~~~~~~~~

**Issue**: Video format not supported

.. code-block:: text

   RuntimeError: Failed to load video: Unsupported video format

**Solutions**:

1. **Install additional codecs**:
   
   .. code-block:: bash
   
      # Install FFmpeg
      conda install ffmpeg -c conda-forge
      
      # Or with apt-get on Ubuntu
      sudo apt-get install ffmpeg

2. **Convert video format**:
   
   .. code-block:: python
   
      import subprocess
      
      def convert_video(input_path, output_path):
           command = [
               'ffmpeg', '-i', input_path,
               '-c:v', 'libx264', '-c:a', 'aac',
               '-strict', 'experimental',
               output_path
           ]
           subprocess.run(command, check=True)

**Issue**: Corrupted video files

.. code-block:: text

   RuntimeError: Error opening video file

**Solutions**:

1. **Check video integrity**:
   
   .. code-block:: python
   
      import cv2
      
      def check_video_integrity(video_path):
           cap = cv2.VideoCapture(video_path)
           if not cap.isOpened():
               return False, "Cannot open video"
           
           frame_count = 0
           while True:
               ret, frame = cap.read()
               if not ret:
                   break
               frame_count += 1
           
           cap.release()
           expected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
           
           return frame_count > 0, f"Read {frame_count} frames"

Training Issues
---------------

Memory Problems
~~~~~~~~~~~~~~

**Issue**: Out of memory errors during training

.. code-block:: text

   RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB

**Solutions**:

1. **Reduce batch size**:
   
   .. code-block:: python
   
      # Reduce from 8 to 4 or 2
      train_loader = DataLoader(dataset, batch_size=4, ...)
   
2. **Use gradient accumulation**:
   
   .. code-block:: python
   
      accumulation_steps = 4
      optimizer.zero_grad()
      
      for i, (videos, labels) in enumerate(train_loader):
           outputs = model(videos)
           loss = criterion(outputs, labels) / accumulation_steps
           loss.backward()
           
           if (i + 1) % accumulation_steps == 0:
               optimizer.step()
               optimizer.zero_grad()

3. **Enable mixed precision training**:
   
   .. code-block:: python
   
      from torch.cuda.amp import GradScaler, autocast
      
      scaler = GradScaler()
      
      for videos, labels in train_loader:
           with autocast():
               outputs = model(videos)
               loss = criterion(outputs, labels)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()

4. **Clear GPU cache**:
   
   .. code-block:: python
   
      import torch
      
      # Add this after each epoch
      torch.cuda.empty_cache()

**Issue**: CPU memory issues

.. code-block:: text

   OSError: [Errno 12] Cannot allocate memory

**Solutions**:

1. **Reduce number of data loader workers**:
   
   .. code-block:: python
   
      train_loader = DataLoader(dataset, num_workers=2, ...)  # Reduce from 8

2. **Use memory mapping for large datasets**:
   
   .. code-block:: python
   
      dataset = MultiViewDataset(path, split='train', mmap_mode='r')

Training Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Very slow training

**Solutions**:

1. **Enable pin_memory for GPU training**:
   
   .. code-block:: python
   
      train_loader = DataLoader(
           dataset, 
           batch_size=8,
           pin_memory=True,  # Faster GPU transfer
           num_workers=4
       )

2. **Use SSD storage for dataset**:
   - Move dataset to SSD if using HDD
   - Consider using RAM disk for small datasets

3. **Optimize data preprocessing**:
   
   .. code-block:: python
   
      # Pre-compute expensive transformations
      class CachedDataset(Dataset):
           def __init__(self, base_dataset):
               self.base_dataset = base_dataset
               self.cache = {}
           
           def __getitem__(self, idx):
               if idx not in self.cache:
                   self.cache[idx] = self.base_dataset[idx]
               return self.cache[idx]

**Issue**: Model not converging

.. code-block:: text

   Training loss not decreasing after several epochs

**Solutions**:

1. **Check learning rate**:
   
   .. code-block:: python
   
      # Try different learning rates
      for lr in [1e-3, 1e-4, 1e-5]:
           optimizer = optim.Adam(model.parameters(), lr=lr)

2. **Add learning rate scheduling**:
   
   .. code-block:: python
   
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(
           optimizer, mode='min', factor=0.5, patience=5
       )

3. **Check data preprocessing**:
   
   .. code-block:: python
   
      # Verify data normalization
      def check_data_stats(loader):
           all_data = []
           for videos, _ in loader:
               all_data.append(videos)
           
           all_data = torch.cat(all_data, dim=0)
           print(f"Mean: {all_data.mean()}")
           print(f"Std: {all_data.std()}")
           print(f"Min: {all_data.min()}")
           print(f"Max: {all_data.max()}")

Inference Issues
----------------

Model Loading Problems
~~~~~~~~~~~~~~~~~~~~~

**Issue**: Model checkpoint loading errors

.. code-block:: text

   RuntimeError: Error(s) in loading state_dict for MVNetwork

**Solutions**:

1. **Check model architecture compatibility**:
   
   .. code-block:: python
   
      # Ensure model architecture matches checkpoint
      model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
      
      # Load with strict=False to ignore missing keys
      model.load_state_dict(checkpoint['state_dict'], strict=False)

2. **Handle DataParallel models**:
   
   .. code-block:: python
   
      # Remove 'module.' prefix if present
      state_dict = checkpoint['state_dict']
      if any(key.startswith('module.') for key in state_dict.keys()):
           state_dict = {key[7:]: value for key, value in state_dict.items()}
       
       model.load_state_dict(state_dict)

**Issue**: Version compatibility problems

.. code-block:: text

   RuntimeError: version_1 <= kMaxSupportedFileFormatVersion

**Solutions**:

1. **Save model with older PyTorch version compatibility**:
   
   .. code-block:: python
   
      # When saving
      torch.save(model.state_dict(), 'model.pth', _use_new_zipfile_serialization=False)

2. **Convert checkpoint format**:
   
   .. code-block:: python
   
      # Load and re-save checkpoint
      checkpoint = torch.load('old_model.pth', map_location='cpu')
      torch.save(checkpoint, 'new_model.pth')

Performance Issues
~~~~~~~~~~~~~~~~~

**Issue**: Slow inference speed

**Solutions**:

1. **Use model.eval() and torch.no_grad()**:
   
   .. code-block:: python
   
      model.eval()
      with torch.no_grad():
           outputs = model(inputs)

2. **Optimize for inference**:
   
   .. code-block:: python
   
      # Use TorchScript for faster inference
      model.eval()
      scripted_model = torch.jit.script(model)
      scripted_model.save('model_scripted.pt')

3. **Batch processing**:
   
   .. code-block:: python
   
      # Process multiple videos at once
      batch_size = 16
      for i in range(0, len(videos), batch_size):
           batch = videos[i:i+batch_size]
           outputs = model(batch)

Interface Issues
---------------

GUI Problems
~~~~~~~~~~~

**Issue**: PyQt5 installation or display issues

.. code-block:: text

   ImportError: No module named 'PyQt5'
   qt.qpa.plugin: Could not load the Qt platform plugin

**Solutions**:

1. **Install PyQt5 correctly**:
   
   .. code-block:: bash
   
      pip install PyQt5
      
      # On Linux, you might need:
      sudo apt-get install python3-pyqt5

2. **Fix Qt platform plugin issues**:
   
   .. code-block:: bash
   
      # Linux
      export QT_QPA_PLATFORM_PLUGIN_PATH=/path/to/qt/plugins
      
      # Windows - reinstall PyQt5
      pip uninstall PyQt5
      pip install PyQt5

**Issue**: Video playback problems in GUI

.. code-block:: text

   Video not playing or black screen in interface

**Solutions**:

1. **Install multimedia codecs**:
   
   .. code-block:: bash
   
      # Linux
      sudo apt-get install ubuntu-restricted-extras
      
      # Windows - install K-Lite Codec Pack

2. **Use alternative video backend**:
   
   .. code-block:: python
   
      # In video_window.py, try different backends
      import cv2
      cv2.setUseOptimized(True)

General Performance Optimization
-------------------------------

System-Level Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **GPU Optimization**:
   
   .. code-block:: python
   
      # Enable CUDA optimizations
      torch.backends.cudnn.benchmark = True  # For fixed input sizes
      torch.backends.cudnn.deterministic = False  # For performance

2. **CPU Optimization**:
   
   .. code-block:: python
   
      # Set optimal thread count
      import torch
      torch.set_num_threads(4)  # Adjust based on your CPU

3. **Memory Usage Monitoring**:
   
   .. code-block:: python
   
      import psutil
      import torch
      
      def print_memory_usage():
           # CPU memory
           memory = psutil.virtual_memory()
           print(f"CPU Memory: {memory.percent}% used")
           
           # GPU memory
           if torch.cuda.is_available():
               memory_used = torch.cuda.memory_allocated() / 1024**3
               memory_total = torch.cuda.max_memory_allocated() / 1024**3
               print(f"GPU Memory: {memory_used:.2f}GB / {memory_total:.2f}GB")

Debugging Tools
--------------

Logging and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   import time
   
   # Setup logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('vars_debug.log'),
           logging.StreamHandler()
       ]
   )
   
   class DebugModel(nn.Module):
       def __init__(self, base_model):
           super().__init__()
           self.base_model = base_model
       
       def forward(self, x):
           logging.info(f"Input shape: {x.shape}")
           start_time = time.time()
           
           output = self.base_model(x)
           
           inference_time = time.time() - start_time
           logging.info(f"Inference time: {inference_time:.4f}s")
           logging.info(f"Output shape: {output.shape}")
           
           return output

Error Reporting
~~~~~~~~~~~~~~

.. code-block:: python

   def create_error_report(error, context=None):
       """Create detailed error report for troubleshooting"""
       
       import traceback
       import platform
       import torch
       
       report = {
           'error_type': type(error).__name__,
           'error_message': str(error),
           'traceback': traceback.format_exc(),
           'system_info': {
               'platform': platform.platform(),
               'python_version': platform.python_version(),
               'pytorch_version': torch.__version__,
               'cuda_available': torch.cuda.is_available(),
               'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
           }
       }
       
       if context:
           report['context'] = context
       
       # Save report
       import json
       with open('error_report.json', 'w') as f:
           json.dump(report, f, indent=2)
       
       return report

Getting Help
-----------

Before Seeking Support
~~~~~~~~~~~~~~~~~~~~~

1. **Check this troubleshooting guide** for common solutions
2. **Verify your environment** matches the requirements
3. **Test with minimal examples** to isolate the issue
4. **Check logs** for detailed error messages
5. **Try the latest version** of dependencies

When Reporting Issues
~~~~~~~~~~~~~~~~~~~

Include the following information:

.. code-block:: text

   **Environment Information:**
   - Operating System: [Windows/Linux/macOS version]
   - Python Version: [3.x.x]
   - PyTorch Version: [x.x.x]
   - CUDA Version: [if using GPU]
   - Hardware: [CPU/GPU specifications]
   
   **Issue Description:**
   - What you were trying to do
   - What happened instead
   - Complete error message
   - Steps to reproduce
   
   **Code Sample:**
   ```python
   # Minimal code that reproduces the issue
   ```

Common Error Patterns
-------------------

Quick Reference
~~~~~~~~~~~~~~

.. list-table:: Common Errors and Solutions
   :header-rows: 1
   :widths: 30 70

   * - Error Pattern
     - Quick Solution
   * - ``CUDA out of memory``
     - Reduce batch size or use gradient accumulation
   * - ``ModuleNotFoundError``
     - Check virtual environment and install missing packages
   * - ``RuntimeError: version_1``
     - Update PyTorch or re-save model checkpoint
   * - ``SSL: CERTIFICATE_VERIFY_FAILED``
     - Update certificates or disable SSL verification
   * - ``ImportError: PyQt5``
     - Install PyQt5 with pip or system package manager
   * - ``FileNotFoundError: dataset``
     - Verify dataset path and file permissions
   * - ``RuntimeError: DataLoader worker``
     - Reduce num_workers or disable multiprocessing
   * - ``Permission denied``
     - Check file permissions or run with appropriate privileges

This troubleshooting guide should help resolve most common issues encountered when working with the VARS system. For persistent problems, consider creating a minimal reproducible example and seeking help from the community.
