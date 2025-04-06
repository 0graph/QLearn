# QLearn - Neural Network Training for QGIS

![QLearn Logo](_static/logo.png)

**QLearn** is a QGIS plugin that allows you to train a UNet based neural network for classification and regression of raster data. 

## Documentation

Full documentation is available at [https://qlearn.readthedocs.io/](https://qlearn.readthedocs.io/).

## Key Features

* **QGIS Integration**: Train and use machine learning models without leaving QGIS
* **Automatic Data Preprocessing**: Automatic alignment, rescaling, reclassifying, and normalization of rasters
* **Testing and Validation**: Automatically splits datasets into training and evaluation sets and provides model evaluation metrics
* **Confidence Filtering**: Optional filtering of predictions based on confidence levels
* **Multiband Support**: Works with any number of input bands and output classes

## Installation

It is suggested that you install QGIS through the OSGeo4W installer, as it simplifies the installation process. You can download it from the 
[OSGeo4W Website](https://trac.osgeo.org/osgeo4w/)

1. Open OSGeo4w Shell
2. Install the necessary dependencies using the following command: `pip3 install torch torchvision`
  * Note: if you did not install QGIS through OSGeo4W installer you may need to manually install torch and torchvision to QGIS's python directory.
3. Open QGIS and navigate to the Plugins menu.
4. Select Manage and Install Plugins.
5. Search for QLearn and click Install Plugin.
6. Restart QGIS to complete the installation.
7. After restarting, you should see the QLearn plugin in the Processing Toolbox.

## Basic Usage

### Training
1. In QGIS, go to Processing Toolbox → QLearn → Training → QLearnTrain
2. Select input/target raster pairs
3. Choose training type (classification or Regression)
4. Set the output model location
5. Click Run to start training

### Prediction
1. In QGIS, go to Processing Toolbox → QLearn → Prediction → QLearnPredict
2. Select the input raster for prediction
3. Choose your trained model (.pth file)
4. (optional) Set the output location
5. Click Run to generate predictions

## Requirements

* QGIS 3.26+ (earlier versions are untested but may work)
* Python with torch and torchvision packages
* OsGeo4W (Recommended)
* Windows 10+ (Linux and MacOS are untested but may work)

## Support

If you encounter issues or have questions:

* Check the [FAQ](https://qlearn.readthedocs.io/en/latest/faq.html)
* Report bugs on the [Issue Tracker](https://github.com/0graph/QLearn/issues)
* Visit the [QLearn Plugin Page](https://plugins.qgis.org/plugins/qlearn/) (coming soon)


