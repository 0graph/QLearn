from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback
from qgis.analysis import QgsAlignRaster
from qgis import processing

from .QLearnDataset import QDataset



class QPreprocessing:

    def __init__(self, 
                 context: QgsProcessingContext, 
                 feedback: QgsProcessingFeedback, 
                 training_rasters: list[QgsRasterLayer] = list(), 
                 target_rasters: list[QgsRasterLayer] = list(), 
                 args: dict = dict()):
        self.context = context                      # Context for QGIS Processing Algorithms
        self.feedback = feedback                    # Feedback Manager for QGIS Processing Algorithms
        self.chunkSize = 256                        # Split Images into Chunks of this size
        self.NODATA = -1                            # NoData Value for rasters
        self.training_rasters = training_rasters    # Training rasters for QLearn
        self.target_rasters = target_rasters        # Target rasters for QLearn

    # Preforms necessary preprocessing steps for QLearn and returns a QDataset which is used by the PyTorch DataLoader
    def process(self)-> QDataset:
        return QDataset()

    # Aligns a training raster and a target raster
    def alignRasters( self,
            training_raster: QgsRasterLayer, 
            target_raster: QgsRasterLayer) -> tuple[bool,QgsRasterLayer,QgsRasterLayer]:
        
        mem_training = "memory:training_raster"
        mem_target = "memory:target_raster"

        self.replace_NODATA(training_raster, mem_training)
        self.replace_NODATA(target_raster, mem_target)

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
    
    # Replaces NODATA values with a predefined 
    def replace_NODATA(self, ras: QgsRasterLayer, outputDestination: str) -> bool:
        pass
    
    # Split raster into chunks
    def gen_chunks(self, ras: QgsRasterLayer):
        pass

    # Generates augmented raster images to enhance training
    def gen_augmentations(self, ras: QgsRasterLayer):
        pass