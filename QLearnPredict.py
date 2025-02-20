import torch
from qgis.core import QgsProcessingFeedback, QgsRasterLayer, QgsProcessingContext, QgsRectangle, QgsRasterBlock, Qgis, QgsRasterDataProvider, QgsDataSourceUri, QgsError
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
        chX, chY = self.preprocessor.calculate_chunks(in_raster)
        raster_data: np.ndarray = in_raster.as_numpy()
        raster_data = raster_data.transpose(0, 2, 1)
        self.feedback.pushInfo(f"Raster Data Shape: {raster_data.shape.__str__()} Data {raster_data}")
        out_raster_data = np.ndarray(shape=(in_raster.width(),in_raster.height()),dtype=raster_data.dtype)
        out_raster_data.fill(1000.0)
        self.feedback.pushInfo(f"InRaster: {in_raster.name()} # Chunks [{chX},{chY}] DataType[{raster_data.dtype.__str__()}]")
    
        self.model.eval()  # Ensure model is in evaluation mode

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        t_iterations = chX * chY

        for iX in range(chX):
            for iY in range(chY):
                if self.feedback.isCanceled():
                    return

                chunk = self.read_chunk(raster_data, iX, iY)

                self.feedback.pushInfo(f"Chunk Shape: {chunk.shape.__str__()} Data {chunk}")

                input_tensor = torch.tensor(chunk.astype(np.float32), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    prediction = torch.argmax(output, dim=1)
                    unique_classes = prediction.unique()
                    self.feedback.pushInfo(f"Unique predicted classes in batch: {unique_classes.cpu().numpy()}")
                    prediction = prediction.squeeze().cpu().numpy() 

                    probabilities = torch.softmax(output, dim=1)
                    max_probs, preds = torch.max(probabilities, dim=1)
                    self.feedback.pushInfo(f"MaxProbs: {max_probs} Preds: {preds}")
                    self.feedback.pushInfo(f"Prediction Shape: {prediction.shape.__str__()} Data {prediction}")

                    x_start, x_end = self.chunkSize * iX, self.chunkSize * (iX + 1)
                    y_start, y_end = self.chunkSize * iY, self.chunkSize * (iY + 1)
                    x_end = min(x_end, in_raster.width())
                    y_end = min(y_end, in_raster.height())
                    
                    out_raster_data[x_start:x_end, y_start:y_end] = prediction[:x_end - x_start, :y_end - y_start]
                
                self.feedback.setProgress((((iX * chY) + iY) / t_iterations)*100)

        # Write the final raster data
        self.write_raster_data(in_raster, out_ras_path, out_raster_data)
        return QgsRasterLayer(out_ras_path)



    def write_raster_data(self, in_raster: QgsRasterLayer, out_raster_path: str, data: np.ndarray) -> bool:
        
        out_raster = QUtils.createSinglebandRaster(
            destination="memory:predicted_raster",
            feedback=self.feedback,
            crs=in_raster.crs(),
            extent=in_raster.extent(),
            width=in_raster.width(),
            height=in_raster.height()
        )

        if out_raster is None or not out_raster.isValid():
            self.feedback.pushWarning("Error: Output raster is not valid!")
            return None

        provider = out_raster.dataProvider()
        self.feedback.pushInfo(f"Provider URI: {provider.dataSourceUri()} Raster SRC: {out_raster.source()} Provider Bands: {provider.bandCount()} band1desc:{provider.bandDescription(1)}")
        
        provider.setEditable(True)
        provider.bandDescription(1)

        if not provider.isValid():
            self.feedback.pushInfo("ERROR: Cannot write raster data, provider is invalid")
            return False
        
        self.feedback.pushInfo(f"DataShape: {data.shape.__str__()} RasterShape: ({provider.xSize()},{provider.ySize()})")
        self.feedback.pushInfo(f"Mean: {data.mean()}, Data: {data}")

        #block = QgsRasterBlock(Qgis.DataType.Float32, data.shape[0], data.shape[1])
        block = provider.block(1,provider.extent(),provider.xSize(),provider.ySize())
        if not block.isValid():
            self.feedback.pushInfo(f"Error: Cannot write raster data, block is invalid")
            return False
        
        
        self.feedback.pushInfo(f"BlockData: {block.width()},{block.height()} - E:{block.isEmpty()} - T:{block.dataType()}")
        
        i = 0
        for xI in range(provider.xSize()):
            for yI in range(provider.ySize()):
                block.setValue(yI, xI, data[xI, yI])

                i+=1
                if(i % 5000 == 0):
                    self.feedback.pushInfo(f"Writing {data[xI, yI]} to block[{xI,yI}] for {i} Result: {block.value(yI, xI)}")
        
        #block.setData(data.tobytes())
        if not block.isValid():
            self.feedback.pushInfo(f"Error: Cannot write raster data, block is invalid 2")
            return False

        if not provider.writeBlock(block,1,0,0):
            self.feedback.pushInfo("ERROR: Cannot write raster data, write operation failed")
            return False
        
        provider.setEditable(False)
        
        self.feedback.pushInfo(f"URI: {provider.dataSourceUri()} Errors:{provider.error().summary()}")

        QUtils.setRasterDestination(out_raster,out_raster_path,self.feedback,self.context)

        return True
        

    def read_chunk(self, data: np.ndarray, chX: int, chY: int) -> np.ndarray:
        sX = chX * self.chunkSize
        sY = chY * self.chunkSize
        eX = sX + self.chunkSize
        eY = sY + self.chunkSize

        # self.feedback.pushInfo(f"Reading Chunk ({chX},{chY}) from array with shape: {data.shape.__str__()}")
        
        # Pad data if needed
        pad_x = max(0, eX - data.shape[1])
        pad_y = max(0, eY - data.shape[2])
        
        if pad_x > 0 or pad_y > 0:
            data = np.pad(data, ((0, 0), (0, pad_x), (0, pad_y)), mode='constant')
        
        return data[:, sX:eX, sY:eY]
        
       
        