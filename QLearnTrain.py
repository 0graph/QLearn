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
        self.device = args.get("DEVICE", torch.device("cpu"))           # CPU or GPU
        self.n_classes = args.get("N_CLASSES", 1)                       # Number of output classes, this should be determined automatically from the target dataset
        self.task = args.get("TRAIN_TYPE", "regression")                # regression or classification
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
            batch_size=args.get("BATCH_SIZE",4),                        # Batch size for processing
            shuffle=False,                                              #
            num_workers=0)                                              # Set to 0 as there are multithreading issues I believe
                                                                        #       
        self.epochs = args.get("EPOCHS",10)                             # Number of epochs to train for
        self.learning_rate = args.get("LEARNING_RATE",1e-3)             # Learning Rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # Optimizer
        self.model_output_location = output_loc                         # Where to save model file
        self.feedback = feedback                                        # For processing algorithm

        # Use different loss criterion depending on segmentation type
        if self.task == "classification": 
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.NODATA)
            self.feedback.pushInfo(f"IgnoreIndex: {self.dataset.NODATA}")
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
                
                self.feedback.pushInfo(f"BEFORERESIZE Output shape: {outputs.shape}, Target shape: {targ_chunk.shape}")
                
                if self.task == "classification":   # (CrossEntropyLoss)
                    targ_chunk = targ_chunk.long()  # Convert to Class labels
                else:                               # Regression (MSELoss)
                    targ_chunk = targ_chunk.float() # Ensure float values
                    

                if targ_chunk.ndim == 4 and targ_chunk.shape[1] == 1:
                    targ_chunk = targ_chunk.squeeze(1)     

                self.feedback.pushInfo(f"AFTERRESIZE Output shape: {outputs.shape}, Target shape: {targ_chunk.shape}")

                self.feedback.pushInfo(f"Sample Output: {outputs[0, :, 100, 100].detach().cpu().numpy()}")
                self.feedback.pushInfo(f"Sample Target: {targ_chunk[0, 100, 100].detach().cpu().numpy()}")

                loss = self.criterion(outputs, targ_chunk)  # Compute loss without masking

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
