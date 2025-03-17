
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.ma as ma
from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback, QgsRasterDataProvider, QgsRectangle, QgsRasterBlock, Qgis
from .QLearnUtils import QUtils, NormalizationParams

from .QRasterNumpy import *


# Class format used by pytorch dataloader
class QDataset(Dataset):
    def __init__(self,
                 training_rasters: list[QgsRasterLayer],
                 target_rasters: list[QgsRasterLayer],
                 context: QgsProcessingContext,
                 feedback: QgsProcessingFeedback,
                 args: dict,
                 checkpoint: dict):
        
        self.training_rasters = training_rasters
        self.target_rasters = target_rasters
        self.context = context
        self.feedback = feedback
        self.chunk_indices = []                                     # Indices of each chunk for each raster in aligned_rasters
        self.aligned_rasters = []                                   # The list of aligned raster filenames
        self.chunkSize = args["CHUNK_SIZE"]                         # Split Images into Chunks of this size
        self.NODATA = args["NODATA"]                                # NoData Value for rasters
        self.bands = 999                                            # Calculated from each training raster, will use the lowest value. initalized to 999 so any amount of bands can be accepted
        self.task = args["TRAIN_TYPE"]                              # regression or classification
        self.normalize_inputs = args["NORMALIZE_INPUTS"]            # weather to normalize the input values in _getitem_
        self.normalize_targets = args["NORMALIZE_TARGETS"]          # weather to normalize the target values in _getitem_
        self.norm_params_train: list[NormalizationParams] = None    # mean and scale values for normalization of training data
        self.norm_params_target: list[NormalizationParams] = None   # mean and scale values for normalization of target data
        self.checkpoint = checkpoint                                # checkpoint dictionary
                                                                    # Eventually using a reduction method for larger rasters like PCA would be ideal
                                                                    # Or filling the ndarray with values that pytorch ignores to preserve the maximum amount of data
        self.do_class_mapping = args["CLASS_REMAPPING"]             # Weather to preform automatic class remapping
        self.class_mapping = {}                                     # the class mapping dictionary { new_class : old_class }
        self.inv_class_mapping = {}                                 # Used for rempapping tensors { old_class : new_class }
        self.NODATA_class_mapping = -100                            # used to set CrossEntropyLoss ignore index


        self.normalize_targets = self.normalize_targets and self.task == "regression" # only normalize targets for regression

        # will overwrite the passed in params
        self.load_checkpoint_data() # load checkpoint data if it exists

        if(len(training_rasters) != len(target_rasters)):
            self.feedback.pushWarning("Error: Length of Input Rasters and Target Rasters does not match")
            return
        
        # Align each pair of rasters and save it to a temporary file if valid
        # additionally calculate the total chunks and normalization values
        for i,(train_ras, targ_ras) in enumerate(zip(training_rasters, target_rasters)):
            self.feedback.pushInfo(f"Raster Set {i}: [Training: {train_ras.name()},Target: {targ_ras.name()}] Bands: {train_ras.bandCount()}")

            success, train_ras_align, targ_ras_align = QUtils.alignRasters(train_ras, targ_ras, i, self.feedback, self.context)

            if(not success or targ_ras.bandCount() > 1):
                self.feedback.pushWarning(f"Error: Could not align rasters {train_ras.name(),targ_ras.name()}")
                continue

            self.bands = min(self.bands, train_ras.bandCount()) # Set band count to lowest of any raster in list
            self.aligned_rasters.append((train_ras_align, targ_ras_align))

            train_ras = QgsRasterLayer(train_ras_align)
            targ_ras = QgsRasterLayer(targ_ras_align)
            chX, chY = QUtils.calculate_chunks(train_ras, self.chunkSize)
            self.chunk_indices.extend([(i, x, y) for x in range(chX) for y in range(chY)])

            # Add Class mappings from aligned rasters
            if self.task == "classification" and self.do_class_mapping:
                self.update_class_mapping(targ_ras.as_numpy())

        
        # Calculate normalization parameters for training data
        self.calc_normalization_params()
            
        # now that we've update the class mapping, insert nodata class mapping at the end so that if the classes start at 0 
        # then it wont have to shift them for the output
        self.add_NODATA_class_mapping()

    # preloads the checkpoint data for retraining before processing the dataset
    def load_checkpoint_data(self):
        if not self.checkpoint: # no checkpoint data (new training)
            return
        
        model_params = self.checkpoint["model_params"]
        self.chunkSize = model_params["out_sz"][0]

        training_params = self.checkpoint["training_params"]
        self.NODATA = training_params["NODATA"]
        self.task = training_params["task_type"]
        self.normalize_inputs = training_params["normalize_inputs"]
        self.do_class_mapping = training_params["do_class_mapping"] 
        self.normalize_targets = training_params["normalize_targets"]
        self.norm_params_train = self.checkpoint["norm_params_train"]
        self.norm_params_target = self.checkpoint["norm_params_target"]
        self.class_mapping = self.checkpoint["class_mapping"]
        self.inv_class_mapping = self.checkpoint["inv_class_mapping"]
        self.NODATA_class_mapping = self.checkpoint["NODATA_class_mapping"]

        # Debugging Statements
        self.feedback.pushInfo(f"Loaded Checkpoint Data: NODATA[{self.NODATA}] TASK[{self.task}] NORMALIZE_INPUTS[{self.normalize_inputs}] DO_CLASS_MAPPING[{self.do_class_mapping}] CHUNKSIZE[{self.chunkSize}]")
        self.feedback.pushInfo(f"Loaded Checkpoint Data: Training Normalization Params: {self.norm_params_train}")
        self.feedback.pushInfo(f"Loaded Checkpoint Data: Target Normalization Params: {self.norm_params_target}")
        self.feedback.pushInfo(f"Loaded Checkpoint Data: Class Mapping: {self.class_mapping}")
        self.feedback.pushInfo(f"Loaded Checkpoint Data: Inverse Class Mapping: {self.inv_class_mapping}")
        self.feedback.pushInfo(f"Loaded Checkpoint Data: NODATA Class Mapping: {self.NODATA_class_mapping}")



    def calc_normalization_params(self):
        if not self.normalize_inputs:
            return
        
        # initialize normalization parameters (if checkpoint is not none then they should be initialized)
        if(self.checkpoint is None):
            self.norm_params_train = [NormalizationParams() for _ in range(self.bands)]
            self.norm_params_target = [NormalizationParams()] # only one target band
        else:
            assert self.norm_params_train is not None and self.norm_params_target is not None, "Normalization parameters must be initialized if loading from checkpoint"
        
        for train_ras, targ_ras in self.aligned_rasters:
            train_ras = QgsRasterLayer(train_ras)
            targ_ras = QgsRasterLayer(targ_ras)

            # Calculate normalization parameters for training data
            data = train_ras.as_numpy(use_masking=True)
            for b in range(min(data.shape[0], self.bands)):
                # calculate mean and scale for each band
                self.norm_params_train[b].update_from_array(data[b])

            # Calculate normalization parameters for target data
            data = targ_ras.as_numpy(use_masking=True)

            self.norm_params_target[0].update_from_array(data)

        self.feedback.pushInfo(f"Training Normalization Params: {self.norm_params_train}")
        self.feedback.pushInfo(f"Target Normalization Params: {self.norm_params_target}")
                
        

    def add_NODATA_class_mapping(self):
        # only want to add class mapping if classification and we're doing class mapping
        if self.task != "classification" or not self.do_class_mapping:
            return


        if self.NODATA not in self.class_mapping.values():
            # add NODATA as the last class 
            self.NODATA_class_mapping = max(self.class_mapping.keys()) + 1
            self.class_mapping[self.NODATA_class_mapping] = self.NODATA

        # update inverse mapping to add NODATA at the end
        self.inv_class_mapping = {cls: i for i, cls in self.class_mapping.items()}

        # Debugging statements
        self.feedback.pushInfo(f"Finalized Class Mapping: {self.class_mapping}")  
        self.feedback.pushInfo(f"Finalized Inverse Class Mapping: {self.inv_class_mapping}")

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        # Get Chunks
        raster_idx, chX, chY = self.chunk_indices[idx]
        train_filename, target_filename = self.aligned_rasters[raster_idx]
        # Get Chunk Data
        training_chunk = self.read_chunk(train_filename, chX, chY)
        target_chunk = self.read_chunk(target_filename, chX, chY)
        # Create Tensors
        training_tensor = torch.tensor(training_chunk, dtype=torch.float32)
        target_tensor = torch.tensor(target_chunk, dtype=torch.float32)

        # Normalize training data
        if self.normalize_inputs:
            training_tensor = QUtils.normalize(training_tensor, self.NODATA, self.norm_params_train, self.feedback)

        if self.task == "regression": # need to normalize regression targets for now to prevent exploding gradients
            target_tensor = QUtils.normalize(target_tensor, self.NODATA, self.norm_params_target, self.feedback)
        else: # convert to long tensor before class remapping
            target_tensor = torch.round(target_tensor).long()
        

        # Return training data and remapped target data
        return training_tensor, self.remap_classes(target_tensor)
    
    # preform class remapping based on dictionary (target raster)
    def remap_classes(self, tensor : torch.tensor) -> torch.tensor:
        if not self.do_class_mapping or self.task != "classification":
            return tensor
        
        np_tensor = tensor.numpy()
        # Create output array with the same shape
        remapped_array = np.full(np_tensor.shape, fill_value=self.NODATA_class_mapping, dtype=np.int64)  # NODATA

        # Apply mapping using inverse mapping
        for old_class, new_class in self.inv_class_mapping.items():
            remapped_array[np_tensor == old_class] = new_class

        return torch.tensor(remapped_array, dtype=torch.int64)
        
    # update class mapping with new unique values
    def update_class_mapping(self, arr : np.array):
        unique_classes = np.unique(arr)

        # Update the class mapping with new classes
        for ucls in unique_classes:
            if ucls not in self.class_mapping.values() and ucls != self.NODATA:
                new_index = len(self.class_mapping)
                self.class_mapping[new_index] = ucls
                self.inv_class_mapping[ucls] = new_index

        self.feedback.pushInfo(f"Updated Class Mapping: {self.class_mapping}")  # Debugging statement

    def read_chunk(self, ras_filename: str, chX: int, chY: int) -> np.ndarray:
        raster = QgsRasterLayer(ras_filename) # Fails on multithread
        raster_band_count = min(self.bands,raster.bandCount())

        
        # Initialize a 3D array with the NODATA value
        data = np.full((raster_band_count, self.chunkSize, self.chunkSize), self.NODATA, dtype=np.float64)

        if not raster.isValid():
            self.feedback.pushWarning(f"ERROR: Issue Reading Raster {ras_filename}")
            return data  # Return empty chunk filled with NODATA
        
        
        # Calculate chunk boundaries
        # TODO: Refactor this
        provider = raster.dataProvider()
        xOffset = chX * self.chunkSize
        yOffset = chY * self.chunkSize
        xRes = raster.rasterUnitsPerPixelX()
        yRes = abs(raster.rasterUnitsPerPixelY())
        xSize = min(self.chunkSize, provider.xSize() - xOffset)
        ySize = min(self.chunkSize, abs(provider.ySize()) - yOffset)
        x_min = raster.extent().xMinimum() + xOffset * xRes
        y_max = raster.extent().yMaximum() - yOffset * yRes
        x_max = x_min + xSize * xRes
        y_min = y_max - ySize * yRes
        chunkBounds = QgsRectangle(x_min, y_min, x_max, y_max)

        # Iterate over each band and extract chunk
        for b in range(1, raster_band_count + 1):
            block: QgsRasterBlock = provider.block(b, chunkBounds, xSize, ySize)
            if not block:
                self.feedback.pushInfo(f"ERROR: Failed to read block for band {b}")
                continue
            
            #self.feedback.pushInfo(f"BLOCK: B[{b}] MB[{raster_band_count}] DT[{block.dataType()}] - V[{block.isValid()}] - S[{block.toString()}]")

            # Set Block's Datatype
            if(not block.convert(Qgis.DataType.Float64)):
                self.feedback.pushWarning(f"Error: Could not convert block's DataType")

            # NumPy create a masked numpy array from the block
            # Note: if the block does not have a NODATA value it will use a default value for making which can cause conflicts
            # Replace block's actual NODATA value with NODATA
            m_block = block.as_numpy(use_masking=False)  

            # Replace NaN values with NODATA
            if np.isnan(m_block).any():
                m_block = np.nan_to_num(m_block, nan=self.NODATA)

            # Apply masking if NODATA value exists
            if block.hasNoDataValue():
                mask = (m_block == block.noDataValue())
                m_block[mask] = self.NODATA # replace block's NODATA with our NODATA

            # self.feedback.pushInfo(f"Block ({b},{chX},{chY}): W:[{block.width()}] H:[{block.height()}] NODATA:[{NoDataVal}] SHP:[{m_block.shape.__str__()}]")
            # Assign block data to the correct slice of the data array
            data[b - 1, :ySize, :xSize] = m_block

        return data