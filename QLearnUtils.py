from qgis.core import QgsRasterLayer, QgsProcessingFeedback, QgsProcessingUtils, QgsProcessingContext, Qgis, QgsRasterFileWriter
from qgis.analysis import QgsAlignRaster
from qgis import processing
import torch
import numpy as np

# Potential solution for normalizing data using online normalization methods
# Source: https://github.com/cerebras/online-normalization

# Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# Using Welford's online algorithm for calculating mean and standard deviation
class NormalizationParams:
    def __init__(self):
        self.n = 0          # count of values seen so far
        self.mean = 0.0     # mean of values seen so far
        self.M2 = 0.0       # sum of squares of differences from mean

    # update based on array
    def update_from_array(self, arr: np.ndarray):
        for x in arr.flatten():
            self.update(x)

    # update mean and std
    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2 

    def get_mean(self):
        return self.mean

    def get_std(self):
        return (self.M2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0
    
    def __str__(self):
        return f"Mean: {self.mean}, Std: {self.get_std()}, N: {self.n}"

    
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

    # calculate chunk shapes with overlap
    # each chunk will have overlap on all sides
    @staticmethod
    def calculate_overlap_chunks(ras: QgsRasterLayer, chunkSize: int, overlap: int) -> list:
        chunks = []
        width = ras.width()
        height = ras.height()
        
        # Calculate step size based on overlap
        effective_size = chunkSize - overlap
        
        # calculate chunk start positions
        for i in range(0, width, effective_size):
            for j in range(0, height, effective_size):
                chunks.append((i, j))
        
        return chunks

    # Generates augmented raster images to enhance training (input only)
    @staticmethod
    def gen_augmentations(ras: QgsRasterLayer):
        pass


    # Source: https://abagen.readthedocs.io/en/stable/user_guide/normalization.html
    # Soruce: https://github.com/rmarkello/abagen/blob/main/abagen/correct.py
    # Robust sigmoid normalization
    @staticmethod
    def sigmoid_normalization(tensor: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
        return 1 / (1 + torch.exp(-(tensor - mean) / scale))
    
    # inverse of the sigmoid normalization
    @staticmethod
    def sigmoid_denormalization(tensor: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
        return mean - scale * torch.log(1 / tensor - 1)
    

    # Denormalizes data values (output only)
    @staticmethod
    def denormalize(tensor: torch.Tensor, NODATA: float, params: list[NormalizationParams], feedback) -> torch.Tensor:
        mask = tensor == NODATA

        if tensor.ndim == 2:
            assert len(params) == 1, "Normalization params must be provided for each band"
            tensor = QUtils.sigmoid_denormalization(tensor, params[0].get_mean(), params[0].get_std())
        elif tensor.ndim == 3:
            assert len(params) == tensor.shape[0], "Normalization params must be provided for each band"
            for i in range(tensor.shape[0]):
                tensor[i] = QUtils.sigmoid_denormalization(tensor[i], params[i].get_mean(), params[i].get_std())
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions - actual shape: {tensor.size()}")
        
        # replace NODATA values after denormalization
        tensor[mask] = NODATA

        #if feedback is not None:
        #    feedback.pushInfo(f"Denormalized Tensor: mean[{tensor.mean()}], std[{tensor.std()}] min[{tensor.min()}], max[{tensor.max()}]")

        return tensor


    # Normalizes data values (input only)
    # Using a sigmoid function to deal with potentially larger values discovered in later training phases
    # NormalizationParams must be precalculated from the entire dataset and provided for each band to ensure accurate normalization
    @staticmethod
    def normalize(tensor: torch.Tensor, NODATA: float, params: list[NormalizationParams], feedback = None) -> torch.Tensor:

        if tensor.ndim == 2:  # Single-band raster

            assert len(params) == 1, "Normalization params must be provided for each band"

            # Mask out NODATA values
            nodataMask = tensor != NODATA 
            valid_values = tensor[nodataMask]

            # If there are any valid values -> normalize
            if valid_values.numel() > 0:  

                # sigmoid normalization
                tensor[nodataMask] = QUtils.sigmoid_normalization(tensor[nodataMask], params[0].get_mean(), params[0].get_std())

        elif tensor.ndim == 3:  # Multi-band raster

            assert len(params) == tensor.shape[0], "Normalization params must be provided for each band"

            for i in range(tensor.shape[0]):  # Normalize each band separately
                # Mask out NODATA values
                nodataMask = tensor[i] != NODATA
                valid_values = tensor[i][nodataMask]

                # If there are any valid values -> normalize
                if valid_values.numel() > 0:
                    
                    # sigmoid normalization
                    tensor[i][nodataMask] = QUtils.sigmoid_normalization(tensor[i][nodataMask], params[i].get_mean(), params[i].get_std())
                    
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions - actual shape: {tensor.size()}")
        
        # test if normalization is working
        #if feedback is not None:
        #    feedback.pushInfo(f"Normalized Tensor: mean[{tensor.mean()}], std[{tensor.std()}] min[{tensor.min()}], max[{tensor.max()}]")

        return tensor

