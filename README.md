## QLearn
A QGIS Plugin allowing for neural network model training and prediction using the UNet Architecture

#### Dependencies
- Torch, TorchVision
  - Close QGIS and open the OSGeo4W shell
    - If you did not install QGIS using OSGeo4W it is reccomended you reinstall with OSGeo4W
  - Run `pip3 install torch torchvision`
  - Once installed the plugin should be able to run

#### Training
- Inputs:
  - The inputs must partially overlap as only the overlapping sections will be used for training
  - Training Raster - An n-band raster (the independent variable)
  - Target Raster - A singleband raster (the dependent variable) -> the wanted class or output value based on the input
- Outputs: A trained pytorch model

#### Prediction
- Inputs:
  - Model - A trained pytorch model
  - Raster - An n-band raster that you want output values predicted for
- Outputs: A raster of predicted values for the input raster based on the trained model.

#### Issues
- `QLearnPreprocessing:`
  - Dosent deal with error values in rasters
- `QLearnTrainingAlgorithm:`
  - Dosent have an easy interface for selecting a pair of input and targets rasters, just two seperate lists
- `Multithreading:` multithreading causes issues with training -> find workaround to allow multithreading.
  - could temporarily save numpy arrays to disk instead of using images directly
- `Regression Normalization:` currently regression targets are normalized, if this is the case a mapping needs to be saved so the predictor can convert the normalized values back to the original ones
- `Class Mappings:` class mappings should be saved in the checkpoint for retraining and for proper remapping when predicting
- `Retraining:` if retraining is done, certain values need to be updated
  - *class mappings:* for classification the class mappings may need to be expanded
  - *normalization mappings:* for regression the normalization may need to be expanded to account for a larger range


#### Testing Needed
- `Input Data Types:` Are input raster data types being converted correctly
- `Invalid Values:` Test raster with invalid/NaN values
- `Regression:` is regression working properly?
- `NODATA:` is training rasters with NODATA valuess working properly?


#### Features
- `Confidence Value:` allow confidence value to be specified for prediction
- `Normalization:` =add option to choose weather you want to normalize the data
- `Data Augementation:` allow option to generate n augmented rasters for training
- `Class Weightings:` Ignore index is not enough if multiple classes should be ignored, allow training weights to be specified for each class for CrossEntropyLoss
- `Confidence Level:` Confidence levels should be able to be specified in predict, and tensor probabilities below that should be set to the specified NODATA value

#### Refactoring
- Move WriteRasterData from QLearnPredict to QLearnUtils
- Possibly make read-chunk in QLearnDataset to be like the one in QLearnPredict -> assumed that train and target rasters have exact same extent which they should
  - could even combine them into one