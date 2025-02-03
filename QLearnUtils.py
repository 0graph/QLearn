from qgis.core import QgsRasterLayer, QgsRasterPipe, QgsProcessingFeedback, QgsRasterFileWriter

class Utils:
    @staticmethod
    def saveRasterToDisk(ras: QgsRasterLayer, filename: str, feedback: QgsProcessingFeedback) -> bool:
        if(not ras.isValid()):
            feedback.pushInfo("Error: Cannot save raster to disk - Raster is not valid")
            return False
        pipe = QgsRasterPipe()
        pipe.set(ras.dataProvider().clone())
        file_writer = QgsRasterFileWriter(filename)
        result = file_writer.writeRaster(pipe, ras.width(), ras.height(), ras.extent(), ras.crs())
        return True
