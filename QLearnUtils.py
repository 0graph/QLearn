from qgis.core import QgsRasterLayer, QgsProcessingFeedback, QgsProcessingContext, Qgis
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
        

    QgisDataType_to_NumPyDataType = {
        Qgis.DataType.Byte: np.byte,
        Qgis.DataType.Float32: np.float32,
        Qgis.DataType.Float64: np.float64,
        Qgis.DataType.UInt16: np.uint16,
        Qgis.DataType.UInt32: np.uint32,
        Qgis.DataType.Int16: np.int16,
        Qgis.DataType.Int32: np.int32,
        Qgis.DataType.UnknownDataType: None
    }

    def dt2np(dataType: Qgis.DataType):
        return QUtils.QgisDataType_to_NumPyDataType.get(dataType, None)
        
