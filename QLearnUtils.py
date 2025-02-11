from qgis.core import QgsRasterLayer, QgsProcessingFeedback, QgsProcessingContext, Qgis, QgsRasterFileWriter, QgsRasterDataProvider
from qgis import processing
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
