from qgis.core import QgsRasterLayer, QgsProcessingFeedback, QgsProcessingContext, Qgis
from qgis import processing

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
        
        
