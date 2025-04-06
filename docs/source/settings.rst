Settings & Features
===================

.. image:: _static/training_menu.png
   :alt: QLearn Prediction Menu

Basic Training Parameters
-------------------------

.. list-table::
   :widths: 20 50 10 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
     - Type
   * - Raster Pairs
     - Input/target raster pairs for training. Each pair consists of an input raster and expected output raster. Use the Eval Only flag for a pair of rasters to exclude the rasters from training and only use them to evaluate the effectivness of the model on unseen data after training ends.
     - N/A
     - List of (input, target, evalOnly)
   * - Number of Epochs
     - Number of complete passes through the training dataset.
     - 10
     - Integer
   * - Learning Rate
     - Controls how much to adjust model weights during optimization.
     - 0.001
     - Float
   * - Training Type
     - Type of task: classification (categorical output) or regression (continuous values).
     - Classification
     - Enum
   * - Input Model
     - Existing model to continue training (optional).
     - None
     - File
   * - Output Model
     - Location to save the trained model.
     - N/A
     - File

Model Architecture Parameters
-----------------------------

.. list-table::
   :widths: 20 50 10 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
     - Command Line Flag
   * - Depth
     - Depth of the UNet model (number of down/up sampling operations). Higher depth helps with learning more complex relationships but increases training time
     - 4
     - --depth, -d
   * - Channels
     - Number of base feature channels in the first UNet layer. Higher values can help with learning more complex relationships but increases training time
     - 64
     - --channels, -c

Data Processing Parameters
--------------------------

.. list-table::
   :widths: 20 50 10 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
     - Command Line Flag
   * - NODATA Value
     - Value to treat as no data in input rasters. This is in addition to the Raster's own NODATA value which is converted to this value before training.
     - -100
     - N/A
   * - Normalize Input
     - Whether to normalize input values to [0,1] range.
     - True
     - N/A
   * - Normalize Targets
     - Whether to normalize target values (regression only). Note: values will be denormalized during prediction if this is 'True'
     - True
     - --normalize_targets, -n
   * - Chunk Size
     - Size of image chunks used for training (e.g., 256×256 pixels).
     - 256
     - --chunk_size, -ch
   * - Rescale Factor
     - Downscale images by this factor (e.g., 0.5 = half size). Downscaling can increase training speed by reducing the number of chunks.
     - 1.0
     - --rescale, -r

Training Process Parameters
---------------------------

.. list-table::
   :widths: 20 50 15 15
   :header-rows: 1

   * - Parameter
     - Description
     - Default
     - Command Line Flag
   * - Batch Size
     - Number of samples processed before model weights are updated. 
     - 16
     - --batch_size, -b
   * - Validation Split
     - Fraction of data used for validation during training.
     - 0.2
     - --validation_split, -v
   * - Class Weights
     - Weighting for different classes (classification only). Useful for ignoring multiple classes, or increasing training sensitivity to classes with small sample sizes.
     - None
     - --weights, -w
   * - Early Stopping Patience
     - Number of epochs with no improvement before training stops.
     - 5
     - --end_patience, -ep
   * - Save Mode
     - Whether to save best model (0) or last model (1).
     - 0
     - --save_mode, -sm
  

Advanced Options
----------------

.. list-table::
   :widths: 20 50 15 15
   :header-rows: 1

   * - Parameter
     - Description
     - Default
     - Command Line Flag
   * - Profiling
     - Enable performance profiling during training. Results will be saved to plugins/QLearn/profile_results
     - False
     - --profile, -p


.. image:: _static/prediction_menu.png
   :alt: QLearn Prediction Menu

Prediction Parameters
----------------------

When using a trained model for prediction, the following parameters are available:

.. list-table::
   :widths: 25 60 15
   :header-rows: 1
   
   * - Parameter
     - Description
     - Default
   * - Input Raster
     - Raster to perform prediction on.
     - N/A
   * - Model
     - Trained model file (.pth) to use for prediction.
     - N/A
   * - Output Raster
     - Location to save the prediction results.
     - N/A
   * - Confidence Level
     - The minimum confidence level at which to output a prediction (classification only). Predictions below this confidence will be overwritten with the NODATA value supplied during training
     - 0.0