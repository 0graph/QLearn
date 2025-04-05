FAQ / Help
==========

Frequently Asked Questions
-------------------------

* Q: How do I install the plugin?
    * A: You can install the plugin from the QGIS Plugin Manager. Search for "QLearn" and click "Install". Additional python libraries may need to be installed manually.
* Q: What versions of QGIS are supported?
    * A: QLearn is designed for QGIS 3.26 and later. Earlier versions may work but are not officially supported.
* What types of data does this work with?
    * A: QLearn is designed for raster data, specifically satellite imagery or other geospatial raster datasets. Vector data can be rasterized using QGIS tools if needed.
* Q: What data do I need?
    * A: You need to have a raster dataset and a corresponding singleband target dataset.
* Q: What preprocessing steps does QLearn perform?
    * A: QLearn performs basic preprocessing such as normalization and alignment of the data. You can also specify additional preprocessing steps in the settings.
* Q: Does this plugin support GPU acceleration?
    * A: No, QLearn is designed for use with the PyTorch library provided by OsGeo4w, which does not support GPU acceleration. 
* Q: Can I use my own neural network architecture?
    * A: Due to the complexity of the plugin, custom architectures are not supported. However, you can modify the existing UNet architecture in the source code if you are comfortable with PyTorch.
* Q: How do I evaluate the model's performance?
    * A: When specifying the model parameters, you can set the 'eval only' flag on a pair of rasters. This will exclude the rasters from training and only use them to evaluate the effectiveness of the model on unseen data after training ends. 



Troubleshooting
--------------

* Error: Failed to Align rasters
    * Solution: Ensure the two rasters have overlapping portions.

* Issue: Model training is slow
    * Solution: Training neural networks can be computationally intensive. Try reducing the number of base channels (-c flag), model depth (-d flag), and rescaling (-r flag) the input data to speed up training.

* Issue: Model predictions are inaccurate
    * Solution: Ensure that your training data is representative of the target data. You may need to collect more training samples, adjust the model parameters, or increase the number of training epochs.


Tips and Tricks
----------------
* Use the ``--rescale`` flag to downscale your input data. This can significantly speed up training time.
* Experiment with different model depths and channel sizes to find the best configuration for your data.
* Use the ``--chunk_size`` flag to adjust the size of the image chunks used for training. Smaller chunks can help with memory management but may slow down training.
   * Note: chunk sizes also affect the context window of the model. Smaller chunks may lead to less context being available for the model to learn from, which can change the results.
* Use the ``--normalize_targets`` flag to normalize your target values (regression only). This can help with training stability and convergence.
* Use the ``--batch_size`` flag to adjust the batch size used during training. Larger batch sizes can speed up training but may require more memory.



Getting Help
-----------

If you encounter issues not covered here:

1. Check the [GitHub repository](https://github.com/0graph/QLearn) for known issues
2. Submit a new issue on the [issue tracker](https://github.com/0graph/QLearn/issues)