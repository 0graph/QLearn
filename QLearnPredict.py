import torch
from qgis.core import QgsProcessingFeedback, QgsRasterLayer, QgsProcessingContext, QgsRectangle, QgsRasterBlock, Qgis
from .QLearnPreprocessing import QPreprocessing
import numpy as np

class QNNPredictor:
    def __init__(self, modelPath: str, chunkSize: int, context: QgsProcessingContext, feedback: QgsProcessingFeedback, args: dict = dict()):
        self.model = torch.load(modelPath)
        self.feedback = feedback
        self.feedback.pushInfo(f"Model: {self.model}")
        self.chunkSize = chunkSize
        self.args = args
        self.preprocessor = QPreprocessing(context,feedback,args)

    def predict(self, in_raster: QgsRasterLayer) -> QgsRasterLayer:
        # chX, chY = self.preprocessor.calculate_chunks(in_raster)
        # out_raster = QgsRasterLayer(in_raster.source())
        raster_data = in_raster.as_numpy()
        self.feedback.pushInfo(f"InRaster {in_raster.name()} - {raster_data}")
        input_tensor = torch.tensor(raster_data).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.softmax(output,dim=1)
            self.feedback.pushInfo("Prediction:")
            self.feedback.pushInfo(prediction.__str__())


    def read_chunk(self, data: np.ndarray, chX: int, chY: int) -> np.ndarray:
        sX = chX * self.chunkSize
        sY = chY * self.chunkSize
        eX = sX + self.chunkSize
        eY = sY + self.chunkSize
        
        # Pad data if needed
        pad_x = max(0, eX - data.shape[1])
        pad_y = max(0, eY - data.shape[2])
        
        if pad_x > 0 or pad_y > 0:
            data = np.pad(data, ((0, 0), (0, pad_x), (0, pad_y)), mode='constant')
        
        return data[:, sX:eX, sY:eY]
        
       
        