Interface Module
================

The interface module provides a PyQt5-based graphical user interface for the VARS system, allowing users to interactively analyze football videos and view foul predictions.

VideoWindow Class
-----------------

The main window class that provides the complete GUI functionality for video analysis.

.. py:class:: VideoWindow(parent=None)
   
   Main application window for the VARS system interface.
   
   :param parent: Parent widget (optional)
   :type parent: QWidget or None
   
   **Attributes:**
   
   * ``show_prediction`` (bool) - Controls whether predictions are displayed
   * ``model`` (MVNetwork) - Loaded neural network model for inference
   * ``softmax`` (nn.Softmax) - Softmax layer for probability computation
   * ``mediaPlayer`` (QMediaPlayer) - Video player component
   * ``video_widget`` (QVideoWidget) - Video display widget
   
   **Key Methods:**
   
   .. py:method:: load_video()
      
      Opens a file dialog to select and load a video file.
      
      Supports common video formats (MP4, AVI, MOV, MKV).
   
   .. py:method:: predict_action()
      
      Performs foul detection on the currently loaded video.
      
      :returns: Prediction results including action class and severity
      :rtype: dict
      
      **Process:**
      
      1. Extracts video frames at 25 FPS
      2. Applies preprocessing transformations
      3. Runs inference through the MVNetwork model
      4. Applies softmax to get probability distributions
      5. Returns predicted action class and offence severity
   
   .. py:method:: play_pause()
      
      Toggles video playback between play and pause states.
   
   .. py:method:: set_position(position)
      
      Sets the video playback position.
      
      :param position: Target position in milliseconds
      :type position: int
   
   .. py:method:: position_changed(position)
      
      Updates UI elements when video position changes.
      
      :param position: Current position in milliseconds
      :type position: int
   
   .. py:method:: duration_changed(duration)
      
      Updates UI elements when video duration is determined.
      
      :param duration: Total video duration in milliseconds
      :type duration: int

Usage Example
-------------

.. code-block:: python

   import sys
   from PyQt5.QtWidgets import QApplication
   from interface.video_window import VideoWindow
   
   # Create application
   app = QApplication(sys.argv)
   
   # Create and show main window
   player = VideoWindow()
   player.showMaximized()
   
   # Run application
   sys.exit(app.exec_())

GUI Components
--------------

The interface includes several key components:

**Video Player**
   - Multi-format video support
   - Standard playback controls (play, pause, seek)
   - Timeline scrubber for navigation
   - Fullscreen mode support

**Prediction Panel**
   - Real-time foul detection results
   - Action class classification (8 categories)
   - Offence severity assessment (4 levels)
   - Confidence scores display

**Control Panel**
   - Video loading interface
   - Prediction toggle controls
   - Analysis parameters adjustment
   - Export functionality for results

Model Integration
-----------------

The interface seamlessly integrates with the MVNetwork model:

.. code-block:: python

   # Model initialization in VideoWindow
   self.model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
   
   # Load pre-trained weights
   load = torch.load('14_model.pth.tar', map_location='cpu')
   self.model.load_state_dict(load['state_dict'])
   self.model.eval()

**Preprocessing Pipeline:**

1. **Frame Extraction**: Videos processed at 25 FPS
2. **Normalization**: Frames normalized using ImageNet statistics
3. **Resizing**: Input resized to 224x224 pixels
4. **Tensor Conversion**: Convert to PyTorch tensors

**Inference Process:**

.. code-block:: python

   # Example inference code
   with torch.no_grad():
       # Forward pass through model
       outputs = self.model(video_tensor)
       
       # Apply softmax for probabilities
       probabilities = self.softmax(outputs)
       
       # Get predictions
       action_pred = torch.argmax(probabilities['action'], dim=1)
       severity_pred = torch.argmax(probabilities['severity'], dim=1)

Configuration
-------------

The interface uses several configuration files:

**Event Dictionary** (``config/classes.py``):

.. code-block:: python

   EVENT_DICTIONARY = {
       'action_class': [
           'No action', 'Throwing', 'Other', 'Tackling', 
           'Standing tackling', 'High leg', 'Holding', 'Pushing'
       ],
       'offence_severity_class': [
           'No offence', 'No card', 'Yellow card', 'Red card'
       ]
   }

**Model Configuration**:

- **Backbone**: MViT_V2_S (Motion Video Transformer)
- **Aggregation**: Attention-based temporal aggregation
- **Input Resolution**: 224x224 pixels
- **Frame Rate**: 25 FPS sampling

Error Handling
--------------

The interface includes robust error handling:

**Video Loading Errors**:
   - Unsupported format warnings
   - Corrupted file detection
   - Missing file notifications

**Model Inference Errors**:
   - GPU/CPU fallback mechanisms
   - Memory management for large videos
   - Invalid input handling

**UI Error Recovery**:
   - Graceful degradation for missing components
   - User-friendly error messages
   - Automatic retry mechanisms

Performance Optimization
------------------------

**Memory Management**:
   - Efficient video frame buffering
   - Model weight caching
   - Garbage collection for large tensors

**Processing Speed**:
   - GPU acceleration when available
   - Batch processing for efficiency
   - Asynchronous inference for responsiveness

**User Experience**:
   - Progressive loading indicators
   - Real-time prediction updates
   - Smooth video playback during analysis

Customization
-------------

The interface can be customized through:

**Theme Configuration**:

.. code-block:: python

   # Custom styling
   self.setStyleSheet("background: #0F0F65;")

**Model Selection**:

.. code-block:: python

   # Different backbone networks
   model = MVNetwork(net_name="r3d_18", agr_type="max")
   model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")

**Display Options**:
   - Prediction overlay styles
   - Confidence threshold settings
   - Color schemes for different action classes

Dependencies
------------

The interface module requires:

- **PyQt5**: GUI framework
- **OpenCV**: Video processing
- **MoviePy**: Video manipulation
- **PyTorch**: Model inference
- **Torchvision**: Video I/O and transformations
- **Pandas**: Data handling
- **NumPy**: Numerical operations

Installation
------------

.. code-block:: bash

   # Install GUI dependencies
   pip install PyQt5 opencv-python moviepy
   
   # Install ML dependencies
   pip install torch torchvision
   
   # Install data processing
   pip install pandas numpy

Troubleshooting
---------------

**Common Issues:**

1. **Video Codec Problems**:
   - Install additional codecs: ``pip install opencv-python-headless``
   - Use VLC backend: Set ``IMAGEIO_FFMPEG_EXE`` environment variable

2. **GUI Display Issues**:
   - Check Qt installation: ``python -c "import PyQt5"``
   - Verify display settings for high-DPI screens

3. **Model Loading Errors**:
   - Ensure model file path is correct
   - Check PyTorch version compatibility
   - Verify sufficient memory for model loading

**Performance Tips:**

- Use GPU when available for faster inference
- Reduce video resolution for real-time processing
- Close other applications to free memory
- Use SSD storage for faster video loading
