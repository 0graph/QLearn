## QLearn
A QGIS Plugin allowing for neural network model training and prediction using the UNet Architecture

#### Dependencies
- PyTorch
  - Close QGIS and open an administrator command prompt
  - Navigate to the QGIS Python Folder
  - Run `C:\Program Files\YourQGISFolder\apps\Python39> .\Scripts\pip.exe install --target .\Lib\site-packages\ torch`

#### Issues
- `QLearnDataset`
  - Dosent take datatype of rasters into account -> all rasters need to be converted to same datatype in read_chunk to be used correctly in pytorch
    - I think this is causing issues with np.frombuffer
- `QLearnPreprocessing`
  - Dosent generate chunks covering the entire image
  - Dosent deal with error values in rasters
- `QLearn_algorithm`
  - Dosent have an easy interface for selecting a pair of input and targets rasters, just two seperate lists