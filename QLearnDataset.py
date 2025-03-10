
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.ma as ma
from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback, QgsProcessingUtils, QgsRasterDataProvider, QgsRectangle, QgsRasterBlock, Qgis
from .QLearnUtils import QUtils


# Class format used by pytorch dataloader
class QDataset(Dataset):
    def __init__(self,
                 training_rasters: list[QgsRasterLayer],
                 target_rasters: list[QgsRasterLayer],
                 context: QgsProcessingContext,
                 feedback: QgsProcessingFeedback,
                 args: dict = dict()):
        
        self.training_raster = training_rasters
        self.target_rasters = target_rasters
        self.context = context
        self.feedback = feedback
        self.chunk_indices = []                                     # Indices of each chunk for each raster in aligned_rasters
        self.aligned_rasters = []                                   # The list of aligned raster filenames
        self.chunkSize = args["CHUNK_SIZE"]                         # Split Images into Chunks of this size
        self.NODATA = args["NODATA"]                                # NoData Value for rasters
        self.bands = args["BANDS"]                                  # Calculated from each training raster, will use the lowest value. 
        self.task = args["TRAIN_TYPE"]                              # regression or classification
                                                                    # Eventually using a reduction method for larger rasters like PCA would be ideal
                                                                    # Or filling the ndarray with values that pytorch ignores to preserve the maximum amount of data
        self.do_class_mapping = args["CLASS_REMAPPING"]             # Weather to preform automatic class remapping
        self.class_mapping = {0 : self.NODATA}                      # the class mapping dictionary { new_class : old_class }
        self.inv_class_mapping = {self.NODATA : 0}                  # Used for rempapping tensors { old_class : new_class }


        if(len(training_rasters) != len(target_rasters)):
            self.feedback.pushWarning("Error: Length of Input Rasters and Target Rasters does not match")
            return
        
        # Align each pair of rasters and save it to a temporary file if valid
        for i,(train_ras, targ_ras) in enumerate(zip(training_rasters, target_rasters)):
            self.feedback.pushInfo(f"Raster Set {i}: [Training: {train_ras.name()},Target: {targ_ras.name()}] Bands: {train_ras.bandCount()}")

            success, train_ras_align, targ_ras_align = QUtils.alignRasters(train_ras, targ_ras, self.feedback)

            if(not success or targ_ras_align.bandCount() > 1):
                self.feedback.pushWarning(f"Error: Could not align rasters {train_ras.name(),targ_ras.name()}")
            else:
                self.bands = min(self.bands, train_ras.bandCount()) # Set band count to lowest of any raster in list
                
                # Save to file so rasters dont exceeed memory capacity
                training_aligned_filename = QgsProcessingUtils.generateTempFilename(f"training_aligned_{i}.tif")
                target_aligned_filename = QgsProcessingUtils.generateTempFilename(f"target_aligned_{i}.tif")
                QUtils.setRasterDestination(train_ras_align, training_aligned_filename,self.feedback,self.context)
                QUtils.setRasterDestination(targ_ras_align,target_aligned_filename,self.feedback,self.context)
                self.aligned_rasters.append((training_aligned_filename, target_aligned_filename))

        # Calculate total number of chunks across all rasters to be used by PyTorch DataLoader
        for i, (train_ras_f, targ_ras_f) in enumerate(self.aligned_rasters):
            train_ras = QgsRasterLayer(train_ras_f)
            targ_ras = QgsRasterLayer(targ_ras_f)
            chX, chY = QUtils.calculate_chunks(train_ras, self.chunkSize)
            self.chunk_indices.extend([(i, x, y) for x in range(chX) for y in range(chY)])

            # Add Class mappings from aligned rasters
            if self.task == "classification" and self.do_class_mapping:
                self.update_class_mapping(targ_ras.as_numpy())

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

        if self.task != "classification":
            target_tensor = QUtils.normalize(target_tensor, self.NODATA)


        # Return normalized training data and remapped target data
        return QUtils.normalize(training_tensor, self.NODATA), self.remap_classes(target_tensor)
    
    # preform class remapping based on dictionary (target raster)
    def remap_classes(self, tensor : torch.tensor) -> torch.tensor:
        if not self.do_class_mapping or self.task != "classification":
            return tensor
        
        np_tensor = tensor.numpy()
        # Create output array with the same shape
        remapped_array = np.full(np_tensor.shape, fill_value=self.NODATA, dtype=np.int64)  # NODATA

        # Apply mapping using inverse mapping
        for old_class, new_class in self.inv_class_mapping.items():
            remapped_array[np_tensor == old_class] = new_class

        return torch.tensor(remapped_array, dtype=torch.int64)
        
    # update class mapping with new unique values
    def update_class_mapping(self, arr : np.array):
        unique_classes = np.unique(arr)

        self.class_mapping = {i: cls for i, cls in enumerate(unique_classes)}
        self.inv_class_mapping = {cls: i for i, cls in enumerate(unique_classes)}

        self.feedback.pushInfo(f"Updated Class Mapping: {self.class_mapping}")  # Debugging statement

    def read_chunk(self, ras_filename: str, chX: int, chY: int) -> np.ndarray:
        raster = QgsRasterLayer(ras_filename)
        raster_band_count = min(self.bands,raster.bandCount())

        
        # Initialize a 3D array with the NODATA value
        data = np.full((raster_band_count, self.chunkSize, self.chunkSize), self.NODATA, dtype=np.float64)

        if not raster.isValid():
            self.feedback.pushWarning(f"ERROR: Issue Reading Raster {ras_filename}")
            return data  # Return empty chunk filled with NODATA
        
        
        # Calculate chunk boundaries
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
            m_block = block.as_numpy(use_masking = True)
            if m_block is None:
                self.feedback.pushWarning("Failed to convert block to numpy array")
                continue

            NoDataVal = block.noDataValue() if block.hasNoDataValue() else None
            if NoDataVal is not None:
                m_block = np.where(m_block == NoDataVal, self.NODATA, m_block)  # Replace NODATA Values
            m_block = np.nan_to_num(m_block, nan=self.NODATA)                   # Replace Invalid Values
            m_block = ma.filled(m_block, self.NODATA)
            
            # self.feedback.pushInfo(f"Block ({b},{chX},{chY}): W:[{block.width()}] H:[{block.height()}] NODATA:[{NoDataVal}] SHP:[{m_block.shape.__str__()}]")
            # Assign block data to the correct slice of the data array
            data[b - 1, :ySize, :xSize] = m_block

        return data