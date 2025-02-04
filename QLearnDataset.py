
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
        
        self.training_rasters = training_rasters
        self.target_rasters = target_rasters
        self.context = context
        self.feedback = feedback
        self.preprocessor = QPreprocessing(self.context,self.feedback, args)
        self.chunk_indices = []                                     # Indices of each chunk for each raster in aligned_rasters
        self.aligned_rasters = []                                   # The list of aligned raster filenames
        self.chunkSize = args.get("CHUNK_SIZE",256)                 # Split Images into Chunks of this size
        self.dataType = args.get("DATA_TYPE",Qgis.DataType.UInt16)  # Data Type Default: UInt16, Will be used to convert training data to correct datatype
        self.NODATA = args.get("NODATA",-1)                         # NoData Value for rasters
        self.bands = args.get("BANDS",999)                          # Calculated from each training raster, will use the lowest value. 
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

        return torch.tensor(training_chunk), torch.tensor(target_chunk)

    def read_chunk(self, ras_filename: str, chX: int, chY: int) -> np.ndarray:
        raster = QgsRasterLayer(ras_filename)
        
        # Initialize a 3D array with the NODATA value
        data = np.full((self.bands, self.chunkSize, self.chunkSize), self.NODATA, dtype=np.float32)

        if not raster.isValid():
            self.feedback.pushWarning(f"ERROR: Issue Reading Raster {ras_filename}")
            return data  # Return empty chunk filled with NODATA
        
        provider = raster.dataProvider()

        # Calculate chunk boundaries
        width = raster.width()
        height = raster.height()
        xOffset = chX * self.chunkSize
        yOffset = chY * self.chunkSize
        xSize = min(self.chunkSize, width - xOffset)
        ySize = min(self.chunkSize, height - yOffset)
        extent = raster.extent()

        chunkBounds = QgsRectangle(
            extent.xMinimum() + xOffset,
            extent.yMinimum() + yOffset,
            extent.xMinimum() + xOffset + xSize,
            extent.yMinimum() + yOffset + ySize
        )

        # Iterate over each band and extract chunk
        for b in range(1, self.bands + 1):
            block: QgsRasterBlock = provider.block(b, chunkBounds, xSize, ySize)
            if not block:
                self.feedback.pushInfo(f"ERROR: Failed to read block for band {b}")
                continue
            
            # NumPy Copy Array From Buffer As the block's datatype
            block_data = np.frombuffer(
                block.data(), 
                dtype=QUtils.QDataType2NumpPy(block.dataType())
            ).reshape((ySize, xSize))

            if block_data is None:
                self.feedback.pushInfo(f"Error: Failed to convert block data for band {b}")
                continue

            # Assign block data to the correct slice of the data array
            data[b - 1, :ySize, :xSize] = block_data

        return data