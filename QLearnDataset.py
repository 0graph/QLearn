
import torch
from torch.utils.data import Dataset
import numpy as np
import numpy.ma as ma
from qgis.core import QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback, QgsProcessingUtils, QgsRasterDataProvider, QgsRectangle, QgsRasterBlock, Qgis
from .QLearnPreprocessing import QPreprocessing
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
        self.preprocessor = QPreprocessing(self.context,self.feedback, args)
        self.chunk_indices = []                                     # Indices of each chunk for each raster in aligned_rasters
        self.aligned_rasters = []                                   # The list of aligned raster filenames
        self.chunkSize = args["CHUNK_SIZE"]                         # Split Images into Chunks of this size
        self.NODATA = args["NODATA"]                                # NoData Value for rasters
        self.bands = args["BANDS"]                                  # Calculated from each training raster, will use the lowest value. 
                                                                    # Eventually using a reduction method for larger rasters like PCA would be ideal
                                                                    # Or filling the ndarray with values that pytorch ignores to preserve the maximum amount of data

        if(len(training_rasters) != len(target_rasters)):
            self.feedback.pushWarning("Error: Length of Input Rasters and Target Rasters does not match")
            return
        
        # Align each pair of rasters and save it to a temporary file if valid
        for i,(train_ras, targ_ras) in enumerate(zip(training_rasters, target_rasters)):
            self.feedback.pushInfo(f"Raster Set {i}: [Training: {train_ras.name()},Target: {targ_ras.name()}] Bands: {train_ras.bandCount()}")

            success, train_ras_align, targ_ras_align = self.preprocessor.alignRasters(train_ras, targ_ras)

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
        for i, (train_ras_f, _) in enumerate(self.aligned_rasters):
            ras = QgsRasterLayer(train_ras_f)
            chX, chY = self.preprocessor.calculate_chunks(ras)
            self.chunk_indices.extend([(i, x, y) for x in range(chX) for y in range(chY)])

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

        return training_tensor, target_tensor

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