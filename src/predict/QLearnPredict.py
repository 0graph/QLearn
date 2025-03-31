import torch
from qgis.core import QgsProcessingFeedback, QgsRasterLayer, QgsProcessingContext, QgsRectangle, QgsRasterBlock, Qgis, QgsRasterDataProvider, QgsDataSourceUri, QgsError
from ..utils.QLearnUtils import QUtils
from ..model.QLearnUNet import *
import numpy as np

class QNNPredictor:
    def __init__(self, modelPath: str, context: QgsProcessingContext, feedback: QgsProcessingFeedback, args: dict = dict()):
        torch.serialization.add_safe_globals([QUNet, QUBlock, QUEncoder, QUDecoder])
        checkpoint = torch.load(modelPath, weights_only=False)
        self.feedback = feedback
        self.args = args
        self.context = context
        self.min_confidence = args["CONFIDENCE"]                                        # Minimum confidence level for predictions (otherwise overwrite with NODATA)
        self.chunkSize = checkpoint["model_params"]["out_sz"][0]                        # Chunk Size
        self.training_params = checkpoint["training_params"]                            # Training Parameters
        self.NODATA = self.training_params["NODATA"]                                    # NODATA Value
        self.task = self.training_params["task_type"]                                   # Task Type (classification or regression)
        self.normalize_inputs = self.training_params["normalize_inputs"]                # Normalize Inputs (based on training)
        self.normalize_targets = self.training_params["normalize_targets"]              # Normalize Targets (based on training - for regression only)
        self.norm_params_train = self.training_params["normalization_params_train"]     # Normalization Parameters for training data
        self.norm_params_target = self.training_params["normalization_params_target"]   # Normalization Parameters for target data
        self.do_class_mapping = self.training_params["do_class_mapping"]                # Whether to remap the classes
        self.class_mapping = self.training_params["class_mapping"]                      # Class Mapping (old class -> new class)
        self.inv_class_mapping = self.training_params["inv_class_mapping"]              # Inverse Class Mapping (new class -> old class)
        self.overlap = args["OVERLAP"]                                                  # Overlap between chunks (# pixels)

        self.setup_model(checkpoint["model_params"], checkpoint["model_states"])        # Setup Model with the same parameters used for training
        # self.feedback.pushInfo(f"Model: {self.model}")
        # self.feedback.pushInfo(f"Initialized Predictor - NODATA[{self.NODATA}] TASK[{self.task}] NORMALIZE_INPUTS[{self.normalize_inputs}] DO_CLASS_MAPPING[{self.do_class_mapping}] CHUNKSIZE[{self.chunkSize}]")

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
        chunks = QUtils.calculate_overlap_chunks(in_raster, self.chunkSize, self.overlap)
        raster_data: np.ndarray = in_raster.as_numpy()
        raster_data = raster_data.transpose(0, 2, 1) # [y, x] -> [x, y]
        width = in_raster.width()
        height = in_raster.height()

        # Create output raster based on input raster and pre-fill with NODATA values
        out_raster_data = np.ndarray(shape=(width,height),dtype=raster_data.dtype)
        out_raster_data.fill(self.NODATA) # Fill with NODATA in case 

        self.feedback.pushInfo(f"InRaster: {in_raster.name()} # Chunks [{len(chunks)}] DataType[{raster_data.dtype.__str__()}] Dimensions[{width},{height}]")
    
        self.model.eval()  # Ensure model is in evaluation mode
        for chunk_i in range(len(chunks)):
            if self.feedback.isCanceled():
                return

            # reads a chunk from an image and uses the trained model to predict an output value
            self.predict_chunk(raster_data,out_raster_data,chunks[chunk_i],width,height)       
            
            # set progress for the chunk
            self.feedback.setProgress((chunk_i/len(chunks))*100)


        # Write the final raster data
        self.write_raster_data(in_raster, out_ras_path, out_raster_data)
        return QgsRasterLayer(out_ras_path)
    

    # predicts a chunk and writes predictions to output
    def predict_chunk(self, raster_data: np.ndarray, out_raster_data: np.ndarray, chunk: tuple, width: int, height: int):
        chunk_data = self.read_chunk(raster_data, chunk)
       
        input_tensor = torch.tensor(chunk_data.astype(np.float32), dtype=torch.float32)
        if self.normalize_inputs:
            input_tensor = QUtils.normalize(input_tensor, self.NODATA, self.norm_params_train, self.feedback)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor) # make predictions using model

            # Get predictions
            if self.task == "classification":

                probabilities = torch.softmax(output, dim=1)
                max_probs, prediction = torch.max(probabilities, dim=1)
                self.feedback.pushInfo(f"Chunk [{chunk}] - Class Counts [{prediction.unique(return_counts=True)}] - Mean Conf [{max_probs.flatten().mean()}]")
                # Write prediction to output data including probabilities
                self.write_model_output(prediction,input_tensor,out_raster_data,chunk,width,height,max_probs)

            else: # regression

                prediction = output # model output values are used directly
                self.feedback.pushInfo(f"Chunk [{chunk}] - Mean Value [{prediction.mean()}]")

                # denormalize if needed
                if self.normalize_targets:
                    prediction_denorm = QUtils.denormalize(
                        prediction.squeeze(), # denormalize expects 2D tensor
                        self.NODATA, self.norm_params_target, 
                        self.feedback)
                    prediction[0,0] = prediction_denorm # replace normalized values with denormalized values prediction is [1,1,chunkSize,chunkSize]

                # Write prediction to output data
                self.write_model_output(prediction,input_tensor,out_raster_data,chunk,width,height)
    

    # reshapes the prediction (output tensor) and overwrites with NODATA values based on input tensor
    def reshape_and_mask_output(self, prediction: torch.tensor, input_tensor: torch.tensor, probabilities: torch.tensor = None) -> torch.tensor:
        # make mask out of NODATA values in the input tensor and rewrite the predictions with NODATA based on the mask
        nodata_mask = (input_tensor == self.NODATA).all(dim=1)

        # for regression expects size [1,1,chunkSize,chunkSize]
        # for classification expects size [1, chunkSize, chunkSize]
        if self.task == "regression":
            nodata_mask = nodata_mask.unsqueeze(0)
        else:
            nodata_mask = nodata_mask.squeeze(dim=1)

        self.feedback.pushInfo(f"mask shape: {nodata_mask.size()}, prediction shape:{prediction.size()}")
        prediction[nodata_mask] = self.NODATA # overwrite predictions with NODATA values

        # convert to correct format (2D array)
        prediction = prediction.squeeze().numpy()
        return prediction

    # writes the prediction output to the correct slice of the output data
    def write_model_output(self, prediction: torch.tensor, input_tensor: torch.tensor ,out_data: np.ndarray, chunk: tuple, width: int, height: int, probabilities: torch.tensor = None):
        
        # overwrite predictions below the minimum confidence level with NODATA values (only for classification)
        if probabilities is not None and self.min_confidence > 0.0:
            confidence_mask = probabilities < self.min_confidence
            prediction[confidence_mask] = self.NODATA

        # prepare prediction values for output
        prediction = self.reshape_and_mask_output(prediction, input_tensor, probabilities)

        overlap_half = self.overlap // 2
        chunk_x, chunk_y = chunk # original coordinates (note: these can be negative)
        
        # valculate valid prediction region (non-overlapping)
        # negative start values must be adjusted and end values must be adjusted to stay within chunk bounds
        valid_pred_start_x = 0 if chunk_x <= 0 else overlap_half
        valid_pred_start_y = 0 if chunk_y <= 0 else overlap_half
        valid_pred_end_x = self.chunkSize if chunk_x + self.chunkSize >= width else self.chunkSize - overlap_half
        valid_pred_end_y = self.chunkSize if chunk_y + self.chunkSize >= height else self.chunkSize - overlap_half
        
        # Calculate output region in target image
        out_start_x = max(0, chunk_x + valid_pred_start_x)
        out_start_y = max(0, chunk_y + valid_pred_start_y)
        out_end_x = min(width, chunk_x + valid_pred_end_x)
        out_end_y = min(height, chunk_y + valid_pred_end_y)
        
        # calculate size of output area
        out_width = out_end_x - out_start_x
        out_height = out_end_y - out_start_y
        
        # calculate start positions in prediction array
        pred_start_x = valid_pred_start_x - min(0, chunk_x)
        pred_start_y = valid_pred_start_y - min(0, chunk_y)
        
        # calculate end positions in prediction array using output area size
        pred_end_x = pred_start_x + out_width
        pred_end_y = pred_start_y + out_height
        
        # fix out of bounds values
        if pred_end_x > prediction.shape[0] or pred_end_y > prediction.shape[1]:
            pred_end_x = min(pred_end_x, prediction.shape[0])
            pred_end_y = min(pred_end_y, prediction.shape[1])
            out_end_x = out_start_x + (pred_end_x - pred_start_x)
            out_end_y = out_start_y + (pred_end_y - pred_start_y)
            
        # get prediction slice
        prediction_slice = prediction[pred_start_x:pred_end_x, pred_start_y:pred_end_y]
        
        # write to output
        self.feedback.pushWarning(f"output shapes: pred={prediction_slice.shape}, out=({out_end_x-out_start_x},{out_end_y-out_start_y})")
        out_data[out_start_x:out_end_x, out_start_y:out_end_y] = prediction_slice
        

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



    # reads a chunk from an image and pad the data as necesssary
    # Note: padding mode could be an optional command line parameter
    def read_chunk(self, data: np.ndarray, chunk: tuple) -> np.ndarray:

        # calculate the padding required for the chunk
        sX = chunk[0]
        sY = chunk[1]
        eX = sX + self.chunkSize
        eY = sY + self.chunkSize

        
        # pad the chunk data
        # padding is added to beginning and/or end depending on chunk position
        pad_xmin = max(0, -sX)  
        pad_xmax = max(0, eX - data.shape[1])
        pad_ymin = max(0, -sY)
        pad_ymax = max(0, eY - data.shape[2])

        padded_data = np.pad(data, ((0, 0), (pad_xmin, pad_xmax), (pad_ymin, pad_ymax)), mode='edge')  
        
        sX_padded = sX + pad_xmin
        sY_padded = sY + pad_ymin
        eX_padded = eX + pad_xmin
        eY_padded = eY + pad_ymin

        # Ensure the slice of data is within the valid bounds of the padded image
        padded_chunk = padded_data[:, sX_padded:eX_padded, sY_padded:eY_padded]

        return padded_chunk
        
       
        