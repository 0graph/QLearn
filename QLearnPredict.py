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
        self.task = args["TASK_TYPE"]
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
        width, height = in_raster.width(), in_raster.height()

        # Create output raster based on input raster and pre-fill with NODATA values
        out_raster_data = np.ndarray(shape=(in_raster.width(),in_raster.height()),dtype=raster_data.dtype).fill(self.NODATA)

        self.feedback.pushInfo(f"InRaster: {in_raster.name()} # Chunks [{chX},{chY}] DataType[{raster_data.dtype.__str__()}] Dimensions[{in_raster.width()},{in_raster.height()}]")
        t_iterations = chX * chY # for setting progress
    
        self.model.eval()  # Ensure model is in evaluation mode
        for iX in range(chX):
            for iY in range(chY):
                if self.feedback.isCanceled():
                    return

                # reads a chunk from an image and uses the trained model to predict an output value
                self.predict_chunk(raster_data,out_raster_data,iX,iY,width,height)       
                
                # set progress for the chunk
                self.feedback.setProgress((((iX * chY) + iY) / t_iterations)*100)

        # Write the final raster data
        self.write_raster_data(in_raster, out_ras_path, out_raster_data)
        return QgsRasterLayer(out_ras_path)
    

    # predicts a chunk and writes predictions to output
    def predict_chunk(self, raster_data: np.ndarray, out_raster_data: np.ndarray, iX: int, iY: int, width: int, height: int):
        chunk = self.read_chunk(raster_data, iX, iY)
        # TODO: Normalize prediction input data if normalization was done in training
        input_tensor = torch.tensor(chunk.astype(np.float32), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = np.ndarray()

            # Get predictions
            if self.task == "classification":

                probabilities = torch.softmax(output, dim=1)
                max_probs, prediction = torch.max(probabilities, dim=1)
                self.feedback.pushInfo(f"Chunk [{iX},{iY}] - Class Counts [{prediction.unique(return_counts=True)}] - Mean Conf [{max_probs.mean().numel()}]")
                # Write prediction to output data including probabilities
                self.write_model_output(prediction,out_raster_data,iX,iY,width,height,max_probs)

            else: # regression

                prediction = output # model output values are used directly
                self.feedback.pushInfo(f"Chunk [{iX},{iY}] - Mean Value [{prediction.mean().numel()}]")
                # Write prediction to output data
                self.write_model_output(prediction,out_raster_data,iX,iY,width,height)
    

    # writes the prediction output to the correct slice of the output data
    def write_model_output(self, prediction: np.ndarray, out_data: np.ndarray, iX: int, iY: int, width: int, height: int, probabilities = None):
        if probabilities is not None:
            pass #TODO: rewrite predictions that do not meet confidence level with NODATA value using probabilities

        # Calculate indices to place data in
        x_start, x_end = self.chunkSize * iX, self.chunkSize * (iX + 1)
        y_start, y_end = self.chunkSize * iY, self.chunkSize * (iY + 1)
        x_end = min(x_end, width)
        y_end = min(y_end, height)
        
        # Save predictions to out_raster
        out_data[x_start:x_end, y_start:y_end] = prediction[:x_end - x_start, :y_end - y_start]

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

        # Debug Statements
        #self.feedback.pushInfo(f"Provider URI: {provider.dataSourceUri()} Raster SRC: {out_raster.source()} Provider Bands: {provider.bandCount()} band1desc:{provider.bandDescription(1)}")
        #self.feedback.pushInfo(f"DataShape: {data.shape.__str__()} RasterShape: ({provider.xSize()},{provider.ySize()})")
        #self.feedback.pushInfo(f"Mean: {data.mean()}, Data: {data}")

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
        
       
        