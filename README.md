## QLearn
A QGIS Plugin allowing for neural network model training and prediction using the UNet Architecture

#### Dependencies
- PyTorch
  - Close QGIS and open an administrator command prompt
  - Navigate to the QGIS Python Folder
  - Run `C:\Program Files\YourQGISFolder\apps\Python39> .\Scripts\pip.exe install --target .\Lib\site-packages\ torch`

#### Issues
- `QLearnDataset`
  - NoData values are not being set/converted for rasters (In Read Chunk they are supposedly working)
  - All input rasters must be converted to same data type
  - All target rasters must be converted to same data type
  - reading chunks from buffer is not reading the correct value - upgrade to QGIS 3.40 and use as_numpy
- `QLearnPreprocessing`
  - ~~Dosent generate chunks covering the entire image~~
  - Dosent deal with error values in rasters
- `QLearn_algorithm`
  - Dosent have an easy interface for selecting a pair of input and targets rasters, just two seperate lists


#### Testing
- `Partial Chunks:` Do partial chunks contain correct data and nodata values in proper spot in returned array from read_chunk in *QLearnDataset* - initial testing shows its working
- `Input Data Types:` Are input raster data types being converted correctly