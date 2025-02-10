import torch
from qgis.core import QgsProcessingFeedback, QgsRasterLayer, QgsProcessingContext, QgsRectangle, QgsRasterBlock, Qgis
from .QLearnPreprocessing import QPreprocessing
from .QLearnUtils import QUtils
import numpy as np

class QNNPredictor:
    def __init__(self, modelPath: str, chunkSize: int, context: QgsProcessingContext, feedback: QgsProcessingFeedback, args: dict = dict()):
        self.model = torch.load(modelPath)
        self.feedback = feedback
        self.feedback.pushInfo(f"Model: {self.model}")
        self.chunkSize = chunkSize
        self.args = args
        self.context = context
        self.preprocessor = QPreprocessing(context,feedback,args)

    def predict(self, in_raster: QgsRasterLayer, out_ras_path: str) -> QgsRasterLayer:
        # chX, chY = self.preprocessor.calculate_chunks(in_raster)
        QUtils.setRasterDestination(in_raster, out_ras_path, self.feedback, self.context)
        out_raster = QgsRasterLayer(out_ras_path)

        chX, chY = self.preprocessor.calculate_chunks(in_raster)
        raster_data = in_raster.as_numpy()
        self.feedback.pushInfo(f"InRaster: {in_raster.name()} # Chunks [{chX},{chY}]")

        self.model.eval()  # Ensure model is in evaluation mode

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for chunk in self.read_chunk(raster_data, chX, chY):
            input_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.argmax(output, dim=1)
                prediction = prediction.squeeze(0).cpu().numpy()

                self.feedback.pushInfo("Prediction:")
                self.feedback.pushInfo(prediction.__str__())

            self.write_raster_data(out_raster, prediction)
        return out_raster


    def write_raster_data(self, raster: QgsRasterLayer, data: np.ndarray, chX: int, chY: int) -> bool:
        if not raster.isValid():
            self.feedback.pushInfo("ERROR: Cannot write raster data, raster is invalid")
            return False
        
        prov = raster.dataProvider()
        sX = self.chunkSize*chX
        sY = self.chunkSize*chY
        szX = min(raster.width() - (sX + self.chunkSize), self.chunkSize)
        szY = min(raster.height() - (sY + self.chunkSize), self.chunkSize)
        self.feedback.pushInfo(f"Writing Chunk to {raster.name()} - Chunk[{chX},{chY}] Coords[{sX},{sY}] Size[{szX},{szY}] Shape[{data.shape.__str__()}]")

        if not prov.write(data,1,szX,szY,sX,sY):
            self.feedback.pushInfo("ERROR: Cannot write raster data, write operation failed")
            return False

        return True
        

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
        
       
        