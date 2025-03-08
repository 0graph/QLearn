from qgis.core import QgsRasterLayer, QgsProcessingFeedback, QgsProcessingContext, Qgis, QgsRasterFileWriter
from qgis.analysis import QgsAlignRaster
from qgis import processing
from torch import optim
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
            target_raster: QgsRasterLayer, feedback: QgsProcessingFeedback) -> tuple[bool,QgsRasterLayer,QgsRasterLayer]:
        
        mem_training = "memory:training_raster"
        mem_target = "memory:target_raster"

        alignRaster = QgsAlignRaster()
        rasters_to_align = [ # Creates in memory rasters for alignment
            QgsAlignRaster.Item(target_raster.source(),mem_target),
            QgsAlignRaster.Item(training_raster.source(),mem_training)
            ]

        alignRaster.setRasters(rasters_to_align)

        # Set Raster to Align to
        alignRaster.setParametersFromRaster(QgsAlignRaster.RasterInfo(training_raster.source()))

        success = alignRaster.checkInputParameters()
        if(not success):
            feedback.pushInfo(alignRaster.errorMessage())
            return False, None, None
        
        # Run Alignment
        success = alignRaster.run()
        if(not success):
            feedback.pushInfo(alignRaster.errorMessage())
            return False, None, None
        
        aligned_training = QgsRasterLayer(mem_training, "Aligned Training Raster")
        aligned_target = QgsRasterLayer(mem_target, "Aligned Target Raster")

        if not aligned_training.isValid() or not aligned_target.isValid():
            feedback.reportError("Error: Failed to load aligned rasters.")
            return False, None, None
        
        return True, aligned_training, aligned_target
    
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

    # Normalizes data values (input only)
    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        pass
