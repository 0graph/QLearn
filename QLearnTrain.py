import torch
from torch.utils.data import DataLoader
from .QLearnUNet import QUNet
from .QLearnDataset import QDataset
import torch.optim as optim
import torch.nn as nn
from qgis.core import QgsProcessingFeedback



class QUNetTrainer:
    def __init__(self, dataset: QDataset, output_loc: str, feedback: QgsProcessingFeedback, args: dict = dict()):
        self.dataset = dataset                                          # Dataset used for DataLoader
        self.device = args["DEVICE"]                                    # CPU or GPU
        self.n_classes = args["N_CLASSES"]                              # Number of output classes, this should be determined automatically from the target dataset
        self.task = args["TRAIN_TYPE"]                                  # regression or classification
        self.epochs = args["EPOCHS"]                                    # Number of epochs to train for
        self.learning_rate = args["LEARNING_RATE"]                      # Learning Rate
        self.NODATA = args["NODATA"]
        self.model_output_location = output_loc                         # Where to save model file
        self.feedback = feedback                                        # For processing algorithm
                                                                        #
        self.model: QUNet = QUNet(                                      # UNet Model Init
            in_channels=dataset.bands,                                  # Number of bands in input image
            base_channels=64,                                           #
            depth=4,                                                    # Depth of UNET, higher depth = longer training but more complex pattern recognition
            num_class=self.n_classes,                                   # Number of classes to generate for output
            retain_dim=True,                                            #
            out_sz=(dataset.chunkSize, dataset.chunkSize)               # Chunk Size
        ).to(self.device)                                               # Set device to be used for processing
                                                                        #
        self.dataloader = DataLoader(                                   # Dataloader
            self.dataset,                                               # Dataset for Dataloader
            batch_size=args["BATCH_SIZE"],                              # Batch size for processing
            shuffle=False,                                              #
            num_workers=0)                                              # Set to 0 as there are multithreading issues I believe
                                                                        #       
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # Optimizer

        # Use different loss criterion depending on segmentation type
        if self.task == "classification": 
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.NODATA)
            self.feedback.pushInfo(f"IgnoreIndex: {self.NODATA}")
        else:  # Regression
            self.criterion = nn.MSELoss()
        

    def validation(self): # Implement this
        pass

    def train(self):
        self.feedback.pushInfo(f"Training Started. {self.epochs} epochs, {self.dataset.__len__()} chunks")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for img_chunk, targ_chunk in self.dataloader:
                img_chunk, targ_chunk = img_chunk.to(self.device), targ_chunk.to(self.device)

                outputs = self.model(img_chunk)
                
                if self.task == "classification":   # (CrossEntropyLoss)
                    targ_chunk = targ_chunk.long()  # Convert to Class labels
                else:                               # Regression (MSELoss)
                    targ_chunk = targ_chunk.float() # Ensure float values
                    

                if targ_chunk.ndim == 4 and targ_chunk.shape[1] == 1:
                    targ_chunk = targ_chunk.squeeze(1)

                try:
                    loss = self.criterion(outputs, targ_chunk)  # Compute loss without masking
                except IndexError as e:
                    self.feedback.pushWarning("Error: the number of classes specified is LESS then the actual number of classes.")
                    self.feedback.setProgressText("Error: See Logs.")
                    self.feedback.cancel()
                    return

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.feedback.setProgressText(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.dataloader):.4f}")
            self.feedback.setProgress(((epoch+1)/self.epochs)*100)
            
            if(self.feedback.isCanceled()):
                self.feedback.pushInfo("Training Cancelled...")
                self.save_model()
                return
        
        self.feedback.pushInfo("Training Finished!")
        self.save_model()
        

    def save_model(self):
        self.feedback.pushInfo(f"Saved model to {self.model_output_location}")
        torch.save(self.model, self.model_output_location)
