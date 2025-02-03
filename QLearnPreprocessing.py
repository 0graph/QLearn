from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback
from qgis.analysis import QgsAlignRaster



class QPreprocessing:

    def __init__(self, context: QgsProcessingContext, feedback: QgsProcessingFeedback):
        self.context = context
        self.feedback = feedback

    @staticmethod
    def alignRasters(
            training_raster: QgsRasterLayer, 
            target_raster: QgsRasterLayer,
            context: QgsProcessingContext, 
            feedback: QgsProcessingFeedback) -> tuple[bool,QgsRasterLayer,QgsRasterLayer]:
        
        alignRaster = QgsAlignRaster()
        rasters_to_align = [
            QgsAlignRaster.Item(target_raster.source(),"memory:align_training"),
            QgsAlignRaster.Item(training_raster.source(),"memory:align_target")
            ]

        alignRaster.setRasters(rasters_to_align)
        alignRaster.setParametersFromRaster(QgsAlignRaster.RasterInfo(training_raster.source()))

        success = alignRaster.checkInputParameters()
        if(not success):
            feedback.pushInfo(alignRaster.errorMessage())
            return False, None, None
        
        success = alignRaster.run()
        if(not success):
            feedback.pushInfo(alignRaster.errorMessage())
            return False, None, None
        
        aligned_training = QgsRasterLayer("memory:align_training", "Aligned Training Raster")
        aligned_target = QgsRasterLayer("memory:align_target", "Aligned Target Raster")

        if not aligned_training.isValid() or not aligned_target.isValid():
            feedback.reportError("Error: Failed to load aligned rasters.")
            return False, None, None
        
        return True, aligned_training, aligned_target
    
    @staticmethod # Implement
    def remove_NODATA(ras: QgsRasterLayer, NODATAVal: int) -> QgsRasterLayer:
        return ras