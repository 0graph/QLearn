Examples
========

Example 1: Water Classification
--------------------------------

**Description:**

This example demonstrates how to use QLearn to detect and classify water bodies in a raster image.
- Learn how to convert vector data to a raster mask
- Learn how to obtain sattelite imagery from directly within QGIS
- Learn how to train a model using the QLearn plugin
- Learn how to make predictions using the trained model

**Data Sources:**

- Sentinel-2 or Landsat 8 satellite imagery (from STAC API Browser)
- `Ontario Hydro Network (OHN) Dataset <https://geohub.lio.gov.on.ca/datasets/mnrf::ontario-hydro-network-ohn-waterbody/about>`_

**Time Required:**

- 10-20 minutes to gather data
- 15-60 minutes to train the model (depending on the size of the dataset and your computer's performance)
- 1-10 minutes to make predictions (depending on the size of the image and your computer's performance)

Gathering The Data (Using STAC API Browser)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open QGIS and create a new project.
2. Download your preferred Waterbody Vector Dataset (e.g. Ontario Hydro Network) and add it to your QGIS project.
3. Open the plugins menu and select Manage and Install Plugins.
4. Search for "STAC API Browser" and "QLearn" and install them.
   - Note: QLearn requires the installation of additional dependencies. `Follow the steps here to install <https://qlearn.readthedocs.io/en/latest/tutorial.html>`_.
5. Open the STAC API Browser plugin and search for "Sentinel-2" or "Landsat 8" to find satellite imagery. Alternatively you can use your own raster data and skip to step 10.
6. Add a filter for cloud cover (<1%) and the desired extent of the search, then click "Search".
   - Ensure that the extent of the search area is within the area covered by the waterbody dataset.

.. image:: _static/stac-filters.png
   :alt: STAC API Browser Filters.
   :width: 400px

7. Choose one or more desired images from the results list and click "View Assets".
8. Select the desired bands (I suggest using B2, B3, B4, and B8 for Sentinel-2) and click "Download the assets".
9. Once the download is complete, add the downloaded raster files to your QGIS project. You can set this up to be done automatically in the STAC API Browser settings.

Merging the Satellite Imagery bands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often, the bands of the satellite imagery will be downloaded as separate files, QLearn requires them to be merged into a single raster file. 
If your satellite imagery is already in a single file, you can skip this section.

.. image:: _static/merge_rasters.png
   :alt: QGIS Merge Rasters tool dialog.
   :width: 400px

10.  Navigate to the Raster menu and select Miscellaneous > Merge.
11.  In the Merge dialog, select the bands you want to merge (e.g. B2, B3, B4, and B8) and choose an output file name.
12.  Click "Run" to merge the bands into a single raster file.

.. image:: _static/merged_raster.png
   :alt: Merged Raster Result.
   :width: 400px

Converting a Vector Dataset to a Raster Mask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Ontario Hydro Network dataset is a vector dataset, containing different types of waterbodies including Lakes, Rivers, Ponds, Beaver Bonds, and more.
Depending on the dataset you are using, you may not need to convert it to a raster mask or perform filtering.

13. Navigate to the Raster menu and select Conversion > Rasterize (Vector to Raster).
    - Input Layer: Select the Ontario Hydro Network vector layer.
    - Fixed value to burn in: **1**
    - Output raster size units: Pixels
        - Note: in certain versions of QGIS this option is broken. Use Georeferenced units with 0.0001 instead.
    - Output raster size: same as the merged raster (10 for Sentinel-2).
    - Output extent: If using multiple rasters, ignore this or set the extent to include all the rasters. Otherwise, set it to the extent of the merged raster.
    - Output file: Choose a location to save the raster mask.
    - Advanced Parameters: 
        - Pre-initialze the raster with fixed values: **0**
14. Click "Run" to create the raster mask.
15. Navigate to Processing Toolbox > Raster Tools > Fill NoData Cells
16. (If Needed) Select the raster mask you just created, set the fill value to 0, and set the output file name. Then click "Run" to fill the NoData cells.

Training the Model
^^^^^^^^^^^^^^^^^^

17. Navigate to Processing Toolbox > QLearn > Training > QLearnTrain
18. Select the merged raster as the input raster and the raster mask as the target.
    - If you have multiple pairs of rasters, you can add them one by one. 
19. Set the training type to "Classification" and select the output model location.
20. Set the number of epochs to 10 and the learning rate to 0.001.
    - If you want a better model you can increase the number of epochs to 50 or more, but this will take longer.
21. Start the training process by clicking the "Run" button.
22. Monitor the training progress in the log window. This could take a while depending on the size of the dataset and your computer's performance.
    - Once the training is complete, the trained model will be saved to the specified location.

Prediction with a trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note: you may want to download and process another image from the STAC API Browser to test the model, 
but you can also use the same image you trained on.
I also suggest clipping the image to be much smaller as prediction can take a while depending on the size of the image and your computer's performance.
You can do this using the "Clip Raster by Extent" tool in QGIS.

23. Navigate to Processing Toolbox > QLearn > Prediction > QLearnPredict
24. 
