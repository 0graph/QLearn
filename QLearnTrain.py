import torch
from torch.utils.data import DataLoader, random_split
from .QLearnUNet import QUNet
from .QLearnDataset import QDataset
import torch.optim as optim
import torch.nn as nn
from qgis.core import QgsProcessingFeedback
from dataclasses import dataclass

@dataclass
class TrainingMetrics:
    loss: float = 0.0
    accuracy: float = 0.0

class QUNetTrainer:
    def __init__(self, dataset: QDataset, output_loc: str, feedback: QgsProcessingFeedback, args: dict = dict()):

        # Set Training Arguments
        self.dataset = dataset                                          # Dataset used for DataLoader
        self.device = args["DEVICE"]                                    # CPU or GPU
        self.n_classes = len(self.dataset.class_mapping)                # Number of output classes, this should be determined automatically from the target dataset
        self.task = args["TRAIN_TYPE"]                                  # regression or classification
        self.epochs = args["EPOCHS"]                                    # Number of epochs to train for
        self.learning_rate = args["LEARNING_RATE"]                      # Learning Rate
        self.NODATA = args["NODATA"]
        self.batch_size = args["BATCH_SIZE"]
        self.val_split = args["VALIDATION_SPLIT"]                       # 0-1 ratio of data used for validation vs data used for training
        self.model_output_location = output_loc                         # Where to save model file
        self.feedback = feedback                                        # For processing algorithm
                                                                        
        # Setup PyTorch Training Objects
        self.setup_model()
        self.setup_dataloaders(self.dataset)
        self.setup_OSL()

        self.feedback.pushInfo(f"Number of Classes: {self.n_classes} detected.")


    # Setup the optimizer, scheduler, and loss function
    def setup_OSL(self) -> None:
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) 
        # Reduce Learning Rate on Plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau( 
            self.optimizer, mode='min', factor=0.1, patience=2)
        # CrossEntropyLosss if Classification, MSELoss otherwise
        self.criterion = (nn.CrossEntropyLoss(ignore_index=self.NODATA) 
                if self.task == "classification" else nn.MSELoss())

    # setup the UNet model parameters
    def setup_model(self) -> None:
        self.model: QUNet = QUNet(                                      # UNet Model Init
            in_channels=self.dataset.bands,                             # Number of bands in input image
            base_channels=64,                                           #
            depth=4,                                                    # Depth of UNET, higher depth = longer training but more complex pattern recognition
            num_class=self.n_classes,                                   # Number of classes to generate for output
            retain_dim=True,                                            #
            out_sz=(self.dataset.chunkSize, self.dataset.chunkSize)     # Chunk Size
        ).to(self.device)                                               # Set device to be used for processing

    # Configure the dataloaders for training and validation based on the validation split
    def setup_dataloaders(self, dataset: QDataset) -> None:
        val_size = int(self.val_split * len(dataset))
        gen = torch.Generator().manual_seed(42) # for reproducible results
        self.train_dataset, self.val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=gen)

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    # Execute a single epoch of training
    def train_epoch(self) -> TrainingMetrics:
        self.model.train()
        metrics = TrainingMetrics(loss=0.0)
        curr_accuracy = None
        total_samples = 0

        for images, targets in self.train_dl:
            
            # training was cancelled -> exit
            if self.checkCancel():
                return


            images, targets = images.to(self.device), targets.to(self.device)
            targets = self.prepare_targets(targets) # Reshape and Convert for CrossEntropyLoss if needed

            # Calculate loss
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            metrics.loss += loss.item()
            if self.task == "classification":
                # Calculate the batch's accuracy and multiply it by the number of items
                metrics.accuracy += self.calculate_pred_accuracy(outputs, targets) * images.size(0)
                total_samples += images.size(0)


        # Finalize the accuracy calculations
        metrics.loss /= len(self.train_dl) # average loss per batch of chunks
        metrics.accuracy /= total_samples # average accuracy per chunk
        return metrics
    
    # Execute a single epoch of validation
    def val_epoch(self):
        self.model.eval()
        metrics = TrainingMetrics(loss=0.0)
        total_samples = 0

        with torch.no_grad():
             for images, targets in self.val_dl:
                
                # training was cancelled -> exit
                if self.checkCancel():
                    return
                
                images, targets = images.to(self.device), targets.to(self.device)
                targets = self.prepare_targets(targets)
                
                # calculate loss
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # calculate metrics
                metrics.loss += loss.item()
                if self.task == "classification":
                    # Calculate the batch's accuracy and multiply it by the number of items
                    metrics.accuracy += self.calculate_pred_accuracy(outputs, targets) * images.size(0)
                    total_samples += images.size(0)

        metrics.loss /= len(self.train_dl) # average loss per batch of chunks
        metrics.accuracy /= total_samples # average accuracy per chunk
        return metrics

    def train(self):
        self.feedback.pushInfo(f"Training started with {len(self.train_dataset)} samples")
        
        for epoch in range(self.epochs):
            
            # Catch interrupt raised by checkCancelled() and force stop training
            try:
                # Preform training and validation for one epoch
                train_metrics = self.train_epoch()
                val_metrics = self.val_epoch()
            except KeyboardInterrupt:
                return

            self.log_progress(epoch,train_metrics,val_metrics)
            self.scheduler.step(val_metrics.loss)
            

        self.feedback.pushInfo("Training Finished!")
        self.save_model()

    # report progress, accuracy, and loss
    def log_progress(self, epoch: int, train_metrics: TrainingMetrics, val_metrics: TrainingMetrics):
        log_msg = f"""
                    Epoch [{epoch+1}/{self.epochs}] - Training Loss: {train_metrics.loss:.4f} - Validation Loss: {val_metrics.loss:.4f}
                    """
        if self.task == "classification":
            log_msg += f" - Training Accuracy: {train_metrics.accuracy:.2%} - Validation Accuracy: {val_metrics.accuracy:.2%}"

        self.feedback.pushInfo(log_msg)
        self.feedback.setProgress((epoch + 1) / self.epochs * 100)

    def checkCancel(self) -> bool:
        if self.feedback.isCanceled():
            self.feedback.pushInfo("Training Cancelled...")
            self.save_model()
            raise KeyboardInterrupt # Raise interrupt so we can catch in outer loop
        return False

    # Prepares target tensors for training
    def prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            targets = targets.long() # For CrossEntropyLoss
        else:
            targets = targets.float() # For MSELoss

        # If 1 batch -> remove batch dimension
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        return targets
        
    # compute accuracy of classification outputs
    def calculate_pred_accuracy(self, outputs: torch.tensor, targets: torch.tensor) -> float:
        mask = targets != self.NODATA # Mask NODATA values to ensure they're not in accuracy calc
        if not mask.any(): # Avoid Divide by 0 if all values are NODATA
            return 0.0
        
        preds = outputs.argmax(dim=1) # the predicted classification
        # returns the percentage of predictions that were equal to the actual value
        return (preds[mask] == targets[mask]).float().mean().item()

    def save_model(self):
        self.feedback.pushInfo(f"Saved model to {self.model_output_location}")
        torch.save(self.model, self.model_output_location)
