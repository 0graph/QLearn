from qgis.core import QgsRasterLayer, QgsProcessingFeedback, QgsProcessingUtils, QgsProcessingContext, Qgis, QgsRasterFileWriter
from qgis.analysis import QgsAlignRaster
from qgis import processing
from torch import optim, tensor
import torch
import numpy as np


class QUtils:
    @staticmethod
    # Source: https://gis.stackexchange.com/questions/416616/feed-an-existing-raster-to-qgis-raster-destination-parameter-in-qgis-processing
    def setRasterDestination(ras: QgsRasterLayer, filename: str, feedback: QgsProcessingFeedback, context: QgsProcessingContext) -> bool:
        alg1_params = {'INPUT': ras,
                        'TARGET_CRS':None,
                        'NODATA':None,
                        'COPY_SUBDATASETS':False,
                        'OPTIONS':'',
                        'EXTRA':'',
                        'DATA_TYPE':0,
                        'OUTPUT':filename}
        
        processing.run("gdal:translate",
                alg1_params,
                is_child_algorithm=True,
                context=context,
                feedback=feedback)
        
    @staticmethod
    def createSinglebandRaster(destination: str, feedback: QgsProcessingFeedback, crs, extent, width, height) -> QgsRasterLayer:
        writer = QgsRasterFileWriter(destination)
        writer.setOutputFormat("GTiff")  # GeoTIFF format

        # Create the raster file
        provider = writer.createOneBandRaster(
            dataType=Qgis.DataType.Float32,
            width=width,
            height=height,
            extent=extent,
            crs=crs
        )

        # Load the newly created raster as a QgsRasterLayer
        raster_layer = QgsRasterLayer(destination, "Output Raster")
        if not raster_layer.isValid():
            feedback.pushWarning("Error: Failed to create singleband raster")
            return None

        return raster_layer
    
    # Aligns a training raster and a target raster
    @staticmethod
    def alignRasters(
            training_raster: QgsRasterLayer, 
            target_raster: QgsRasterLayer, index: int, feedback: QgsProcessingFeedback, context: QgsProcessingContext) -> tuple[bool,str,str]:
        
        training_aligned_filename = QgsProcessingUtils.generateTempFilename(f"training_aligned_{index}.tif")
        target_aligned_filename = QgsProcessingUtils.generateTempFilename(f"target_aligned_{index}.tif")

        alignRaster = QgsAlignRaster()
        rasters_to_align = [ # Creates in memory rasters for alignment
            QgsAlignRaster.Item(target_raster.source(),target_aligned_filename),
            QgsAlignRaster.Item(training_raster.source(),training_aligned_filename)
            ]

        alignRaster.setRasters(rasters_to_align)

        # Set Raster to Align to
        alignRaster.setParametersFromRaster(QgsAlignRaster.RasterInfo(training_raster.source()))

        success = alignRaster.checkInputParameters()
        if(not success):
            feedback.pushInfo(f"AlignRaster - InputParameterError: {alignRaster.errorMessage()}")
            return False, None, None
        
        # Run Alignment
        success = alignRaster.run()
        if(not success):
            feedback.pushInfo(f"AlignRaster - Failed to complete algorithm: {alignRaster.errorMessage()}")
            return False, None, None
        
        aligned_training = QgsRasterLayer(training_aligned_filename, "Aligned Training Raster")
        aligned_target = QgsRasterLayer(target_aligned_filename, "Aligned Target Raster")

        if not aligned_training.isValid() or not aligned_target.isValid():
            feedback.reportError("Error: Failed to load aligned rasters.")
            return False, None, None

        # Save to file so rasters dont exceeed memory capacity
        #QUtils.setRasterDestination(aligned_training, training_aligned_filename,feedback,context)
        #QUtils.setRasterDestination(aligned_target,target_aligned_filename,feedback,context)
        
        return True, training_aligned_filename, target_aligned_filename
    
    # calculate number of chunks in raster
    @staticmethod
    def calculate_chunks(ras: QgsRasterLayer, chunkSize: int) -> tuple[int, int]:
        width = ras.width()
        height = ras.height()

        # +1 to Account for partial chunks
        chunksX = (width // chunkSize) + 1 
        chunksY = (height // chunkSize) + 1
        return chunksX, chunksY

    # Generates augmented raster images to enhance training (input only)
    @staticmethod
    def gen_augmentations(ras: QgsRasterLayer):
        pass

    import torch

    # Normalizes data values (input only)
    @staticmethod
    def normalize(tensor: torch.Tensor, NODATA: float) -> torch.Tensor:
        if tensor.ndim == 2:  # Single-band raster
            # Mask out NODATA values
            nodataMask = tensor != NODATA 
            valid_values = tensor[nodataMask]

            # If there are any valid values -> normalize
            if valid_values.numel() > 0:  
                min_val, max_val = valid_values.min(), valid_values.max()
                tensor[nodataMask] = (tensor[nodataMask] - min_val) / (max_val - min_val)

        elif tensor.ndim == 3:  # Multi-band raster
            for i in range(tensor.shape[0]):  # Normalize each band separately
                # Mask out NODATA values
                nodataMask = tensor[i] != NODATA
                valid_values = tensor[i][nodataMask]

                # If there are any valid values -> normalize
                if valid_values.numel() > 0:
                    min_val, max_val = valid_values.min(), valid_values.max()
                    tensor[i][nodataMask] = (tensor[i][nodataMask] - min_val) / (max_val - min_val)
                    
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions")

        return tensor

