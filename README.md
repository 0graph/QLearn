# QLearn
A QGIS Plugin allowing for neural network model training and prediction using the UNet Architecture

### Dependencies
- Torch, TorchVision
  - Close QGIS and open the OSGeo4W shell
    - If you did not install QGIS using OSGeo4W it is reccomended you reinstall with OSGeo4W
  - Run `pip3 install torch torchvision`
  - Once installed the plugin should be able to run

### Training
- **Inputs:**
  - The inputs must partially overlap as only the overlapping sections will be used for training
  - Training Raster - An n-band raster (the independent variable)
  - Target Raster - A singleband raster (the dependent variable) -> the wanted class or output value based on the input
- **Outputs:** A trained pytorch model

### Prediction
- **Inputs:**
  - Model - A trained pytorch model
  - Raster - An n-band raster that you want output values predicted for
- **Outputs:** A raster of predicted values for the input raster based on the trained model.
: 

### Issues
- `QLearnTrainingAlgorithm:`
  - Dosent have an easy interface for selecting a pair of input and targets rasters, just two seperate lists
- `Multithreading:` multithreading causes issues with training -> find workaround to allow multithreading.
  - could temporarily save numpy arrays to disk instead of using images directly
- `Retraining:` if retraining is done, certain values need to be updated
  - *class mappings:* 
    - for training classification the class mappings may need to be expanded
      - this should be able to be done after QDataset is initialized 
    - for prediction classes may should be unmapped back to thier original values
- `Type Conversions:` check if there is any issues with classes when converting or comparing floats and fix using a tolerance
- `Prediction Output Display:` fix prediction outputs defaulting to min and max of 0 for displaying values
- `Classification NODATA outputs:` seems like partial chunks have a very high possibility of bad predictions due to nodata remapping


#### Testing Needed
- `Invalid Values:` Test raster with invalid/NaN values
- `Regression:` Is retraining working for regression
- `NODATA:` 
  - is training rasters with NODATA valuess working properly?


#### Features
- `Data Augementation:` allow option to generate n augmented rasters for training
- `Class Weightings:` Ignore index is not enough if multiple classes should be ignored, allow training weights to be specified for each class for CrossEntropyLoss
- `Confidence Level:` Confidence levels should be able to be specified in predict, and tensor probabilities below that should be set to the specified NODATA value
- `Save Best Model:` To avoid overfitting, after a certain amount of time without improvement the best model can be saved
- `Custom Parameters:` allow the user to specify custom input parameters like [learning rate, depth, dropout, etc.]
- `Custom Validation Dataset:` allow the user to specify a completly seperate validation dataset instead of using randomly selected chunks to ensure validation loss is not biased or misrepresented.

### Refactoring
- Move WriteRasterData from QLearnPredict to QLearnUtils
- Possibly make read-chunk in QLearnDataset to be like the one in QLearnPredict -> assumed that train and target rasters have exact same extent/pixel size/dimensions which they should
  - could even combine them into one
- `Normalization Params:` refactor to allow entire chunk (array) to be used for calculations at once instead of value-by-value

  #### **QLearnDataset Initalization:** lots of repeted array creation and dataset looping that could be fixed
  ```
  # Proposed Strucutre:

  loop (training & target):
    Align()
    CalcChunks()
    AdjustClassMapping()
    GetRasterData()
    CalcNormalizationParams()

    loop (chunks):
      preprocessAndSave() ->
        Normalize()
        Remap()
        Save()

  ```
