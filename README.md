## QLearn
A QGIS Plugin allowing for neural network model training and prediction using the UNet Architecture

#### Dependencies
- PyTorch
  - Close QGIS and open an administrator command prompt
  - Navigate to the QGIS Python Folder
  - Run `C:\Program Files\YourQGISFolder\apps\PythonVersion> .\Scripts\pip.exe install --target .\Lib\site-packages\ torch`

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
- `QLearnTrain/QLearnPredict:`
  - Fix raster edges having garbage prediction results
    - Copy the edge by the convolution mask size
- `Cancelling:` should be able to cancel in between chunks not just in between epochs
- `Batch Size:` barch size specified in QLearnTrainingAlgorithm.py is causing issues with not all training data being used in QLearnTrain.py (is 16)


#### Testing Needed
- `Input Data Types:` Are input raster data types being converted correctly
- `Invalid Values:` Test raster with invalid/NaN values
- `Population Density:` By CMA -> Regression


#### Features
- `Confidence Value:` allow confidence value to be specified for prediction
- `Normalization:` allow option to normalize raster values for training and prediction
- `Data Augementation:` allow option to generate n augmented rasters for training
- `Class Weightings:` Ignore index is not enough if multiple classes should be ignored, allow training weights to be specified for each class for CrossEntropyLoss
- `Retraining` a pytorch model file can be loaded along with new rasters to continue training

#### Refactoring
- Combine QLearnPreprocessing and QLearnUtils into QLearnUtils, additionally the class should be fully static
- Move WriteRasterData from QLearnPredict to QLearnUtils
  - Write entire block at once instead of pixel by pixel for speed
- Possibly make read-chunk in QLearnDataset to be like the one in QLearnPredict -> assumed that train and target rasters have exact same extent which they should
  - could even combine them into one