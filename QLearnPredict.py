import torch
from qgis.core import QgsProcessingFeedback, QgsRasterLayer, QgsProcessingContext, QgsRectangle, QgsRasterBlock, Qgis, QgsRasterDataProvider, QgsDataSourceUri, QgsError
from .QLearnUtils import QUtils
from .QLearnUNet import *
import numpy as np

class QNNPredictor:
    def __init__(self, modelPath: str, context: QgsProcessingContext, feedback: QgsProcessingFeedback, args: dict = dict()):
        torch.serialization.add_safe_globals([QUNet, QUBlock, QUEncoder, QUDecoder])
        checkpoint = torch.load(modelPath, weights_only=False)
        
        self.setup_model(checkpoint["model_params"], checkpoint["model_states"])

        self.feedback = feedback
        self.feedback.pushInfo(f"Model: {self.model}")
        self.chunkSize = checkpoint["model_params"]["out_sz"][0]
        self.NODATA = args["NODATA"]
        self.args = args
        self.context = context

    def setup_model(self, m_params: dict, model_state_dict: dict) -> None:

        self.model: QUNet = QUNet(                                      # UNet Model Init
            in_channels=m_params["in_channels"],                             # Number of bands in input image
            base_channels=m_params["base_channels"],                                           #
            depth=m_params["depth"],                                                    # Depth of UNET, higher depth = longer training but more complex pattern recognition
            num_class=m_params["num_class"],                                   # Number of classes to generate for output
            retain_dim=m_params["retain_dim"],                                            #
            out_sz=m_params["out_sz"]     # Chunk Size
        ) 

        self.model.load_state_dict(model_state_dict)

    def predict(self, in_raster: QgsRasterLayer, out_ras_path: str) -> QgsRasterLayer:

        # Get input raster data
        chX, chY = QUtils.calculate_chunks(in_raster, self.chunkSize)
        raster_data: np.ndarray = in_raster.as_numpy()
        raster_data = raster_data.transpose(0, 2, 1) # [y, x] -> [x, y]

        #self.feedback.pushInfo(f"Raster Data Shape: {raster_data.shape.__str__()} Data {raster_data}")

        # Create output raster based on input raster
        out_raster_data = np.ndarray(shape=(in_raster.width(),in_raster.height()),dtype=raster_data.dtype)
        out_raster_data.fill(1000.0) # should be filled with chosen nodata value

        self.feedback.pushInfo(f"InRaster: {in_raster.name()} # Chunks [{chX},{chY}] DataType[{raster_data.dtype.__str__()}] Dimensions[{in_raster.width()},{in_raster.height()}]")
    
        self.model.eval()  # Ensure model is in evaluation mode

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        t_iterations = chX * chY

        for iX in range(chX):
            for iY in range(chY):
                if self.feedback.isCanceled():
                    return

                chunk = self.read_chunk(raster_data, iX, iY)
                
                # Debug
                #self.feedback.pushInfo(f"Chunk Shape: {chunk.shape.__str__()} Data {chunk}")

                input_tensor = torch.tensor(chunk.astype(np.float32), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = self.model(input_tensor)

                    # Get predicted values
                    prediction = torch.argmax(output, dim=1)
                    prediction = prediction.squeeze().cpu().numpy() 
                    probabilities = torch.softmax(output, dim=1)
                    max_probs, preds = torch.max(probabilities, dim=1)

                    # Print out sample of data for debugging
                    self.feedback.pushInfo(f"Unique Classes: {preds.unique()} \nProbabilities: {max_probs.flatten()[:20]} \nPrediction: {preds.flatten()[:20]}")
                    # self.feedback.pushInfo(f"Prediction Shape: {prediction.shape.__str__()} Data {prediction}")

                    # Calculate indices to place data in
                    x_start, x_end = self.chunkSize * iX, self.chunkSize * (iX + 1)
                    y_start, y_end = self.chunkSize * iY, self.chunkSize * (iY + 1)
                    x_end = min(x_end, in_raster.width())
                    y_end = min(y_end, in_raster.height())
                    
                    # Save predictions to out_raster
                    out_raster_data[x_start:x_end, y_start:y_end] = prediction[:x_end - x_start, :y_end - y_start]
                
                # set progress for the chunk
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

        if out_raster is None or not out_raster.isValid() or not out_raster.dataProvider().isValid():
            self.feedback.pushWarning("Error: Output raster is not valid!")
            return None

        provider = out_raster.dataProvider()

        self.feedback.pushInfo(f"Provider URI: {provider.dataSourceUri()} Raster SRC: {out_raster.source()} Provider Bands: {provider.bandCount()} band1desc:{provider.bandDescription(1)}")
        
        self.feedback.pushInfo(f"DataShape: {data.shape.__str__()} RasterShape: ({provider.xSize()},{provider.ySize()})")
        self.feedback.pushInfo(f"Mean: {data.mean()}, Data: {data}")

        block = provider.block(1,provider.extent(),provider.xSize(),provider.ySize())
        
        if not block.isValid():
            self.feedback.pushInfo(f"Error: Cannot write raster data, block is invalid")
            return False
        
        #self.feedback.pushInfo(f"BlockData: {block.width()},{block.height()} - E:{block.isEmpty()} - T:{block.dataType()}")
        
        # Write predicted data
        provider.setEditable(True)
        out_raster.setCrs(in_raster.crs()) # Make sure CRS of out-raster matches
        blockData = data.astype(np.float32).transpose(1, 0) # Convert to correct output format, and transpose [x, y] -> [y, x] for output
        block.setData(blockData.tobytes())  # set the block data using the bytes of the formatted data

        if not provider.writeBlock(block,1,0,0):
            self.feedback.pushInfo("ERROR: Cannot write raster data, write operation failed")
            return False
        
        provider.setEditable(False)

        # Save the raster
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
        
       
        