
About the Project
=================


Introduction
------------
Classification and Segmentation are integral to geospatial analysis, 
being used heavily across the field from land cover classification to crop yield prediction. 
Although these types of analysis can be done in many ways, neural networks are increasingly being used 
due to their ability to recognize complex patterns and relationships in spatial data. 
Unfortunately, designing a neural network and encoding input data for it is a complex task. Requiring programming experience, 
knowledge of neural networks, and knowledge of how to encode the type of data being input into the neural network.


QLearn simplifies this task by allowing users to train a neural network model for segmentation and classification 
of any type of raster geospatial data using a GUI interface from within QGIS.
Since this plugin will be directly integrated with QGIS it means that GIS analysis involving neural network training 
can be done without the need for code or external tools. 
Advantages of this approach include simplifying and speeding up analysis workflows, 
and improving the accessibility of neural network training.

Methods
-------

Preprocessing
^^^^^^^^^^^^^
The first step in the QLearn workflow is to preprocess the input data. 
The QGIS preprocessing workflow aims to simplify the process of preparing data for training by allowing the user to perform a number of common preprocessing steps.
Steps denoted with a * are optional and based on the settings chosen by the user.

.. image:: _static/preprocessing-chart.svg
   :alt: A flowchart showing the preprocessing steps for QLearn. Includes steps for aligning, rescaling, normalizing, reclassifying, chunking, and saving the data to disk.
   :width: 600px
   :align: center

1. **Alignment & Rescaling:** When working with raster-based training data, it is important to ensure that the input data is aligned with the target data.
    With geospatial data, this means that the rasters must have the same coordinate reference system (CRS), pixel size, and extent.
    QLearn will automatically align the input data to the target data, and rescale the target data to match the input data.
    Additionally, as QgsAlignRasters provides the ability to rescale the data, any user specified rescaling will be done in this step.
    This step simplifies the time consuming process of alignment and rescaling, making it easier to prepare the data for training.

2. **Calculating Chunks:** The next step is to calculate the chunks of data that will be used for training. 
    This is done by dividing the input data into smaller chunks of a specified size. 
    This is important for training neural networks, as it allows the model to learn from smaller portions of the data at a time, 
    which can help with convergence and stability during training. 
    Additionally, raster data is often too large to fit into memory all at once, so chunking the data allows for more efficient processing.
    Optionally, the user can manually specify a chunk size to use for training.


Target (mask) Raster Processing
...............................

3. **(Classification Only) Calculate Class Mappings:** Sometimes when working with classification data, the classes may not be continuous from 0 to N, 
    which is a requirement for the loss function used by QLearn (CrossEntropyLoss).
    In this case, the classes will automatically be reclassified to be continuous from 0 to N + 1, 
    where N is the number of classes in the input data +1 for the NODATA class.

Input & Target Raster Processing
.................................

4. **Calculate Normalization Parameters:** The next step is to calculate the normalization parameters per-band for the input data. 
    This step is important for training neural networks, as it helps to stabilize the training process and improve convergence.
    The user can also specify whether or not to normalize the input data.
    Normalization is done using `Welford's Online Algorithm <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_ 
    as the possibility of retraining and the large size of the data makes it impractical to calculate the mean and standard deviation in a single pass.

5. **(Optional) Normalization & Reclassification:** Based on the user settings, the input and target data will be normalized based on the previous calculations.
    Additionally, the target data will be reclassified based on the calculated class mappings
    Normalization is done using sigmoid normalization as the possibility of data outside the previously calculated min/max values during retraining 
    makes it impractical to use min-max normalization. This means that newly encountered data will not be lost during retraining.
    The user can also specify whether or not to normalize the input and target data.

6. **Saving:** The final step in the preprocessing workflow is to split the processed rasters into chunks and save them to disk.
    This is important as GIS datasets are often too large to fit into memory all at once, 
    and preprocessing and saving the data avoids having to normalize and reclassify the data again during training.

Training
^^^^^^^^^^



