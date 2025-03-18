import torch
from qgis.core import QgsProcessingFeedback, QgsRasterLayer, QgsProcessingContext, QgsRectangle, QgsRasterBlock, Qgis, QgsRasterDataProvider, QgsDataSourceUri, QgsError
from .QLearnUtils import QUtils
from .QLearnUNet import *
import numpy as np

class QNNPredictor:
    def __init__(self, modelPath: str, context: QgsProcessingContext, feedback: QgsProcessingFeedback, args: dict = dict()):
        torch.serialization.add_safe_globals([QUNet, QUBlock, QUEncoder, QUDecoder])
        checkpoint = torch.load(modelPath, weights_only=False)
        self.feedback = feedback
        self.args = args
        self.context = context
        self.min_confidence = args["CONFIDENCE"]
        self.chunkSize = checkpoint["model_params"]["out_sz"][0]
        self.training_params = checkpoint["training_params"]
        self.NODATA = self.training_params["NODATA"]
        self.task = self.training_params["task_type"]
        self.normalize_inputs = self.training_params["normalize_inputs"]
        self.normalize_targets = self.training_params["normalize_targets"]
        self.norm_params_train = self.training_params["normalization_params_train"]
        self.norm_params_target = self.training_params["normalization_params_target"]
        self.do_class_mapping = self.training_params["do_class_mapping"]
        self.class_mapping = self.training_params["class_mapping"]
        self.inv_class_mapping = self.training_params["inv_class_mapping"]

        self.setup_model(checkpoint["model_params"], checkpoint["model_states"])
        self.feedback.pushInfo(f"Model: {self.model}")
        self.feedback.pushInfo(f"Initialized Predictor - NODATA[{self.NODATA}] TASK[{self.task}] NORMALIZE_INPUTS[{self.normalize_inputs}] DO_CLASS_MAPPING[{self.do_class_mapping}] CHUNKSIZE[{self.chunkSize}]")

    def setup_model(self, m_params: dict, model_state_dict: dict) -> None:

        self.model: QUNet = QUNet(                           # UNet Model Init
            in_channels=m_params["in_channels"],             # Number of bands in input image
            base_channels=m_params["base_channels"],         #
            depth=m_params["depth"],                         # Depth of UNET, higher depth = longer training but more complex pattern recognition
            num_class=m_params["num_class"],                 # Number of classes to generate for output
            retain_dim=m_params["retain_dim"],               #
            out_sz=m_params["out_sz"]                        # Chunk Size
        ) 

        self.model.load_state_dict(model_state_dict)

    def predict(self, in_raster: QgsRasterLayer, out_ras_path: str) -> QgsRasterLayer:

        # Get input raster data
        chX, chY = QUtils.calculate_chunks(in_raster, self.chunkSize)
        raster_data: np.ndarray = in_raster.as_numpy()
        raster_data = raster_data.transpose(0, 2, 1) # [y, x] -> [x, y]
        width = in_raster.width()
        height = in_raster.height()

        # Create output raster based on input raster and pre-fill with NODATA values
        out_raster_data = np.ndarray(shape=(width,height),dtype=raster_data.dtype)
        out_raster_data.fill(self.NODATA) # Fill with NODATA in case 

        self.feedback.pushInfo(f"InRaster: {in_raster.name()} # Chunks [{chX},{chY}] DataType[{raster_data.dtype.__str__()}] Dimensions[{width},{height}]")
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
       
        input_tensor = torch.tensor(chunk.astype(np.float32), dtype=torch.float32)
        if self.normalize_inputs:
            input_tensor = QUtils.normalize(input_tensor, self.NODATA, self.norm_params_train, self.feedback)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor) # make predictions using model

            # Get predictions
            if self.task == "classification":

                probabilities = torch.softmax(output, dim=1)
                max_probs, prediction = torch.max(probabilities, dim=1)
                self.feedback.pushInfo(f"Chunk [{iX},{iY}] - Class Counts [{prediction.unique(return_counts=True)}] - Mean Conf [{max_probs.flatten().mean()}]")
                # Write prediction to output data including probabilities
                self.write_model_output(prediction,input_tensor,out_raster_data,iX,iY,width,height,max_probs)

            else: # regression

                prediction = output # model output values are used directly
                self.feedback.pushInfo(f"Chunk [{iX},{iY}] - Mean Value [{prediction.mean()}]")

                # denormalize if needed
                if self.normalize_targets:
                    prediction_denorm = QUtils.denormalize(
                        prediction.squeeze(), # denormalize expects 2D tensor
                        self.NODATA, self.norm_params_target, 
                        self.feedback)
                    prediction[0,0] = prediction_denorm # replace normalized values with denormalized values

                # Write prediction to output data
                self.write_model_output(prediction,input_tensor,out_raster_data,iX,iY,width,height)
    

    # writes the prediction output to the correct slice of the output data
    def write_model_output(self, prediction: torch.tensor, input_tensor: torch.tensor ,out_data: np.ndarray, iX: int, iY: int, width: int, height: int, probabilities: torch.tensor = None):
        
        # overwrite predictions below the minimum confidence level with NODATA values (only for classification)
        if probabilities is not None and self.min_confidence > 0.0:
            confidence_mask = probabilities < self.min_confidence
            prediction[confidence_mask] = self.NODATA

        # make mask out of NODATA values in the input tensor and rewrite the predictions with NODATA based on the mask
        nodata_mask = (input_tensor == self.NODATA).all(dim=1)

        # for regression expects size [1,1,chunkSize,chunkSize] for classification expects size [1, chunkSize, chunkSize]
        if self.task == "regression":
            nodata_mask = nodata_mask.unsqueeze(0)
        else:
            nodata_mask = nodata_mask.squeeze(dim=1)

        self.feedback.pushInfo(f"mask shape: {nodata_mask.size()}, prediction shape:{prediction.size()}")
        prediction[nodata_mask] = self.NODATA

        prediction = prediction.squeeze().numpy() # convert to correct format

        # Calculate indices to place data in
        x_start = self.chunkSize * iX
        y_start = self.chunkSize * iY
        x_end = min(x_start + self.chunkSize, width)
        y_end = min(y_start + self.chunkSize, height)

        # interpolate the edges to remove edge artefacts
        #prediction = self.interpolate_edges(prediction)
        
        # Save predictions to out_raster
        out_data[x_start:x_end, y_start:y_end] = prediction[:x_end - x_start, :y_end - y_start]

    # interpolates the edges of the chunk to remove edge artefacts
    # the edges are interpolated using a kernel of size 3x1 and a mean filter
    def interpolate_edges(self, data: np.ndarray):
        lidx = 0 # lower index
        uidx = data.shape[0] - 1 # upper index
        # top
        for i in range(data[0, :].size):
            from_idx = max(lidx, i-1)
            to_idx = min(uidx, i+1)
            mean = data[3, from_idx:to_idx].mean()
            data[0, i] = mean
            data[1, i] = mean
            data[2, i] = mean

        # bottom
        for i in range(data[-1, :].size):
            from_idx = max(lidx, i-1)
            to_idx = min(uidx, i+1)
            mean = data[-4, from_idx:to_idx].mean()
            data[-1, i] = mean
            data[-2, i] = mean
            data[-3, i] = mean

        # left
        for i in range(data[:, 0].size):
            from_idx = max(lidx, i-1)
            to_idx = min(uidx, i+1)
            mean = data[from_idx:to_idx, 3].mean()
            data[i, 0] = mean
            data[i, 1] = mean
            data[i, 2] = mean

        # right
        for i in range(data[:, -1].size):  
            from_idx = max(lidx, i-1)
            to_idx = min(uidx, i+1)  
            mean = data[from_idx:to_idx, -4].mean()
            data[i, -1] = mean
            data[i, -2] = mean
            data[i, -3] = mean

        return data

    def write_raster_data(self, in_raster: QgsRasterLayer, out_raster_path: str, data: np.ndarray) -> bool:
        

        out_raster = QUtils.createSinglebandRaster(
            destination=out_raster_path,
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
        
       
        