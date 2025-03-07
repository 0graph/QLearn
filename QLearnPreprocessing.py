from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback
from qgis.analysis import QgsAlignRaster
import numpy as np



# A class used to preprocess rasters for training and prediction
class QPreprocessing:

    def __init__(self, 
                 context: QgsProcessingContext, 
                 feedback: QgsProcessingFeedback, 
                 args: dict = dict()):
        self.context = context                      # Context for QGIS Processing Algorithms
        self.feedback = feedback                    # Feedback Manager for QGIS Processing Algorithms
        self.chunkSize = args["CHUNK_SIZE"]         # Split Images into Chunks of this size
        self.NODATA = args["NODATA"]               # NoData Value for rasters

    # Aligns a training raster and a target raster
    def alignRasters( self,
            training_raster: QgsRasterLayer, 
            target_raster: QgsRasterLayer) -> tuple[bool,QgsRasterLayer,QgsRasterLayer]:
        
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
            self.feedback.pushInfo(alignRaster.errorMessage())
            return False, None, None
        
        # Run Alignment
        success = alignRaster.run()
        if(not success):
            self.feedback.pushInfo(alignRaster.errorMessage())
            return False, None, None
        
        aligned_training = QgsRasterLayer(mem_training, "Aligned Training Raster")
        aligned_target = QgsRasterLayer(mem_target, "Aligned Target Raster")

        if not aligned_training.isValid() or not aligned_target.isValid():
            self.feedback.reportError("Error: Failed to load aligned rasters.")
            return False, None, None
        
        return True, aligned_training, aligned_target
    
    # calculate number of chunks in raster
    def calculate_chunks(self, ras: QgsRasterLayer) -> tuple[int, int]:
        width = ras.width()
        height = ras.height()

        # +1 to Account for partial chunks
        chunksX = (width // self.chunkSize) + 1 
        chunksY = (height // self.chunkSize) + 1
        return chunksX, chunksY

    # Generates augmented raster images to enhance training (input only)
    def gen_augmentations(self, ras: QgsRasterLayer):
        pass

    # Normalizes data values (input only)
    def normalize(self, arr: np.ndarray) -> np.ndarray:
        pass
