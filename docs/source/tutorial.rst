Tutorial
========

Getting Started with QLearn
---------------------------

This tutorial will walk you through the basic steps of using QLearn for both training a model and making predictions with an existing model.

Installation
------------

It is suggested that you install QGIS through the OSGeo4W installer, as it simplifies the installation process. You can download it from the 
`OSGeo4W Website <https://trac.osgeo.org/osgeo4w/>`_


1. Open OSGeo4w Shell
2. Install the necessary dependencies using the following command: ``pip3 install torch torchvision``
3. Open QGIS and navigate to the Plugins menu.
4. Select Manage and Install Plugins.
5. Search for QLearn and click Install Plugin.
6. Restart QGIS to complete the installation.
7. After restarting, you should see the QLearn plugin in the Processing Toolbox.

Basic Training Workflow
-----------------------
This section will guide you through the basic workflow of training a classification model using QLearn.
For a more in-depth tutorial, follow along with `one of the Examples <https://qlearn.readthedocs.io/en/latest/examples.html>`_.

**Training**

1. **Prepare Your Data**: Ensure your raster data is in a format supported by QGIS (e.g. GeoTIFF). You will also need a corresponding mask file for training.
2. **Open the Training Plugin**: In QGIS, go to the Processing Toolbox and navigate to QLearn > Training > QLearnTrain
3. **Select Your Data**: Choose the input raster and mask files (I suggest choosing 1 pair to start).
4. **Configure Training Parameters**: Set the training type to 'classification' and select an output model location
5. **Start Training**: Click the Run button to start training. You can monitor the progress in the log window.

**Prediction**

1. **Open the Prediction Plugin**: In QGIS, go to the Processing Toolbox and navigate to QLearn > Prediction > QLearnPredict
2. **Select Your Data**: Choose the input raster file you want to predict on.
3. **Select Your Model**: Choose the trained model file you want to use for predictions.
4. **Configure Prediction Parameters**: Set the output location for the predicted raster.
5. **Start Prediction**: Click the Run button to start the prediction process. The predicted raster will be saved to the specified location.

Once the prediction has completed, the predicted raster will be added to your QGIS project. You can visualize the results using the QGIS styling options.


