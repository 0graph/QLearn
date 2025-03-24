
import torch, os, json
from torch.utils.data import Dataset
import numpy as np
import numpy.ma as ma
from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback, QgsRectangle, QgsRasterBlock, Qgis
from .QLearnUtils import QUtils, NormalizationParams

from .QRasterNumpy import *


# class used by PyTorch DataLoader
class QDataLoader(Dataset):
    def __init__(self, chunk_indicies: list, temp_dir: str):
        self.chunk_indices = chunk_indicies
        self.temp_dir = temp_dir

    # load the preprocessed data from a file
    def load_preprocessed_data(self, i: int, chX: int, chY: int, is_target: bool) -> torch.tensor:
        return torch.load(os.path.join(self.temp_dir, QDataset.make_filename(i, chX, chY, is_target)))

    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        # Get Chunks
        raster_idx, chX, chY = self.chunk_indices[idx]
        train_data = self.load_preprocessed_data(raster_idx, chX, chY, False)
        target_data = self.load_preprocessed_data(raster_idx, chX, chY, True)

        return train_data, target_data


# Class for loading and preprocessing the dataset
class QDataset():
    def __init__(self,
                 raster_pairs: list[tuple[QgsRasterLayer, QgsRasterLayer]],
                 context: QgsProcessingContext,
                 feedback: QgsProcessingFeedback,
                 args: dict,
                 checkpoint: dict):
        
        self.raster_pairs = json.loads(raster_pairs) # data is in json string format "[[r1,r2],[r3,r4],...]"
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
        
        # Note: could save numpy arrays to file after preprocessing and class mapping to be used by dataloader
                                                                    # Or filling the ndarray with values that pytorch ignores to preserve the maximum amount of data
        self.do_class_mapping = args["CLASS_REMAPPING"]             # Weather to preform automatic class remapping
        self.class_mapping = {}                                     # the class mapping dictionary { new_class : old_class } # note: could be a list
        self.inv_class_mapping = {}                                 # Used for rempapping tensors { old_class : new_class }
        self.NODATA_class_mapping = -100                            # used to set CrossEntropyLoss ignore index
        self.temp_dir = None                                        # temporary directory to store preprocessed data

        self.make_temp_dir()                                        # create a temporary directory to store the preprocessed data
        self.normalize_targets = self.normalize_targets and self.task == "regression" # only normalize targets for regression

        # will overwrite the passed in params
        self.load_checkpoint_data() # load checkpoint data if it exists

        
        # Align each pair of rasters and save it to a temporary file if valid
        # additionally calculate the total chunks and normalization values
        for i,(train_src, targ_src) in enumerate(self.raster_pairs):
            train_ras = QgsRasterLayer(train_src)
            targ_ras = QgsRasterLayer(targ_src)

            self.feedback.pushInfo(f"Raster Set {i}: [Training: {train_ras.name()},Target: {targ_ras.name()}] Bands: {train_ras.bandCount()}")

            train_ras_align, targ_ras_align = QUtils.alignRasters(train_ras, targ_ras, i, self.feedback, self.context)

            assert train_ras_align is not None and targ_ras_align is not None, f"Error: Could not align rasters {train_ras.name(),targ_ras.name()}"
            assert targ_ras.bandCount() == 1, f"Error: Target Raster has more than 1 band {targ_ras.name()}"

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

        self.preprocess_and_save() # preprocess and save the data to a file

        # Create PyTorch Dataset to be used by DataLoader
        self.PyTorchDataset = QDataLoader(self.chunk_indices, self.temp_dir)

        assert len(self.chunk_indices) > 0, "Error: No Chunks Found"
        assert len(self.aligned_rasters) > 0, "Error: No Aligned Rasters Found"
        assert self.bands > 0, "Error: No Bands Found"
        assert len(self.norm_params_train) == self.bands, "Error: Normalization Parameters for Training Data not initialized properly"

    # make a temporary directory in the plugin folder to store the preprocessed data
    def make_temp_dir(self) -> str:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # create a temporary directory to store the preprocessed data
        self.temp_dir = os.path.join(dir_path, "temp")
        self.clear_temp_dir()
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.feedback.pushInfo(f"Created Temporary Directory: {self.temp_dir}")
        return self.temp_dir

    # clear the temporary directory
    def clear_temp_dir(self):
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))

    # preprocess each file in the dataset and save the numpy array to a file
    # this saves time and repeted computation during the training loop
    def preprocess_and_save(self):
        for i, chX, chY in self.chunk_indices:
            # Get Chunks
            train_filename, target_filename = self.aligned_rasters[i]
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

            target_tensor = self.remap_classes(target_tensor)
            
            # Save the preprocessed data to a file
            self.save_preprocessed_data(i, chX, chY, False, training_tensor)
            self.save_preprocessed_data(i, chX, chY, True, target_tensor)
    
    # consistent filename format for saving / loading preprocessed data
    @staticmethod
    def make_filename(i: int, chX: int, chY: int, is_target: bool) -> str:
        return f"{i}_{chX}_{chY}_{'target' if is_target else 'train'}.pt"

    # save the preprocessed data to a file
    def save_preprocessed_data(self, i: int, chX: int, chY: int, is_target: bool, data: torch.tensor):
        torch.save(data, os.path.join(self.temp_dir, self.make_filename(i, chX, chY, is_target)))

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
        self.norm_params_train = training_params["normalization_params_train"]
        self.norm_params_target = training_params["normalization_params_target"]
        self.class_mapping = training_params["class_mapping"]
        self.inv_class_mapping = training_params["inv_class_mapping"]
        self.NODATA_class_mapping = training_params["NODATA_CLASS_MAPPING"]

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
        # self.feedback.pushInfo(f"Finalized Class Mapping: {self.class_mapping}")  
        # self.feedback.pushInfo(f"Finalized Inverse Class Mapping: {self.inv_class_mapping}")
    
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

        # self.feedback.pushInfo(f"Updated Class Mapping: {self.class_mapping}")  # Debugging statement

    # calculate the chunk bounds for a given chunk index
    # based on the extent of the raster and the chunk size
    def calc_chunk_bounds(self, raster: QgsRasterLayer, chX: int, chY: int) -> QgsRectangle:
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
        return QgsRectangle(x_min, y_min, x_max, y_max), xSize, ySize
    

    # read a block of data from a raster file and mask out NODATA values
    def read_block_data(self, block: QgsRasterBlock, b: int, chunkBounds: QgsRectangle, xSize: int, ySize: int) -> np.ndarray:
        if not block:
            self.feedback.pushInfo(f"ERROR: Failed to read block for band {b}")
            return None
        
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
        
        return m_block

    def read_chunk(self, ras_filename: str, chX: int, chY: int) -> np.ndarray:
        raster = QgsRasterLayer(ras_filename) # Fails on multithread
        raster_band_count = min(self.bands,raster.bandCount())

        
        # Initialize a 3D array with the NODATA value
        data = np.full((raster_band_count, self.chunkSize, self.chunkSize), self.NODATA, dtype=np.float64)

        if not raster.isValid():
            self.feedback.pushWarning(f"ERROR: Issue Reading Raster {ras_filename}")
            return data  # Return empty chunk filled with NODATA
        
        # Calculate chunk boundaries
        chunkBounds, xSize, ySize = self.calc_chunk_bounds(raster, chX, chY)
        

        provider = raster.dataProvider()
        # Iterate over each band and extract chunk
        for b in range(1, raster_band_count + 1):
            block: QgsRasterBlock = provider.block(b, chunkBounds, xSize, ySize)

            # Read block data
            m_block = self.read_block_data(block, b, chunkBounds, xSize, ySize)

            if m_block is None:
                continue

            # self.feedback.pushInfo(f"Block ({b},{chX},{chY}): W:[{block.width()}] H:[{block.height()}] NODATA:[{NoDataVal}] SHP:[{m_block.shape.__str__()}]")
            # Assign block data to the correct slice of the data array
            data[b - 1, :ySize, :xSize] = m_block

        return data