import torch
from torch.utils.data import DataLoader, random_split
from .QLearnUNet import QUNet
from .QLearnDataset import QDataset
import torch.optim as optim
import torch.nn as nn
from qgis.core import QgsProcessingFeedback
from dataclasses import dataclass
import os

@dataclass
class TrainingMetrics:
    loss: float = 0.0
    accuracy: float = 0.0

class QUNetTrainer:
    def __init__(self, dataset: QDataset, output_loc: str, feedback: QgsProcessingFeedback, args: dict = dict(), curr_model_path: str = ""):

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

        self.try_loading_model(curr_model_path)

        self.feedback.pushInfo(f"Number of Classes: {self.n_classes} detected.")


    # try loading states of current model and optimizers to continue training
    def try_loading_model(self, curr_model_path: str):
        self.feedback.pushInfo(f"loading saved model from: {curr_model_path}")
        # no model to load
        if curr_model_path == "":
            return

        # try loading model states
        if os.path.isfile(curr_model_path):
            try:
                checkpoint = torch.load(curr_model_path)
                self.model.load_state_dict(checkpoint["model_states"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                
                # Debug, can remove later
                self.feedback.pushInfo(f"Model State: {self.model.state_dict()}")
                self.feedback.pushInfo(f"Optimizer State: {self.optimizer.state_dict()}")
                self.feedback.pushInfo(f"Scheduler State: {self.scheduler.state_dict()}")
            except Exception as e:
                self.feedback.pushInfo(f"Exception: {e} - failed to load model from: {curr_model_path} - data is invalid.")
        else:
            self.feedback.pushInfo(f"Failed to load model from: {curr_model_path}. starting from scratch.")

    # Setup the optimizer, scheduler, and loss function
    def setup_OSL(self) -> None:
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) 
        # Reduce Learning Rate on Plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau( 
            self.optimizer, mode='min', factor=0.1, patience=4,min_lr=1e-6)
        # CrossEntropyLosss if Classification, MSELoss otherwise
        self.criterion = (nn.CrossEntropyLoss(ignore_index=self.NODATA) 
                if self.task == "classification" else nn.MSELoss())

    # setup the UNet model parameters
    def setup_model(self) -> None:
        self.mbase_channels=64
        self.mdepth=4
        self.mretain_dim=True

        self.model: QUNet = QUNet(                                      # UNet Model Init
            in_channels=self.dataset.bands,                             # Number of bands in input image
            base_channels=self.mbase_channels,                                           #
            depth=self.mdepth,                                                    # Depth of UNET, higher depth = longer training but more complex pattern recognition
            num_class=self.n_classes,                                   # Number of classes to generate for output
            retain_dim=self.mretain_dim,                                            #
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
        total_loss = 0.0
        total_correct = 0
        total_valid = 0

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate metrics
            total_loss += loss.item()
            if self.task == "classification":
               correct, valid = self.calculate_pred_accuracy(outputs, targets)
               total_correct += correct
               total_valid += valid


        # Finalize the accuracy calculations
        metrics.loss = total_loss / len(self.train_dl) # average loss per batch of chunks
        if self.task == "classification":
            metrics.accuracy = total_correct / total_valid if total_valid > 0 else 0.0

        return metrics
    
    # Execute a single epoch of validation
    def val_epoch(self):
        self.model.eval()
        metrics = TrainingMetrics(loss=0.0)
        total_loss = 0.0
        total_correct = 0
        total_valid = 0

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
                total_loss += loss.item()
                if self.task == "classification":
                    correct, valid = self.calculate_pred_accuracy(outputs, targets)
                    total_correct += correct
                    total_valid += valid

        # Finalize the accuracy calculations
        metrics.loss = total_loss / len(self.val_dl) # average loss per batch of chunks
        if self.task == "classification":
            metrics.accuracy = total_correct / total_valid if total_valid > 0 else 0.0

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
        
    # compute correct and valid pixels
    def calculate_pred_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
        mask = targets != self.NODATA  # Mask NODATA values
        valid_pixels = mask.sum().item()
        
        # Early exit
        if valid_pixels == 0:
            return 0, 0
        
        preds = outputs.argmax(dim=1)
        correct = (preds[mask] == targets[mask]).sum().item()
        return correct, valid_pixels

    def save_model(self):
        checkpoint = {
            "model_params": {
                "in_channels": self.dataset.bands,
                "base_channels": self.mbase_channels,
                "depth": self.mdepth,
                "num_class": self.n_classes,
                "retain_dim": self.mretain_dim,
                "out_sz": (self.dataset.chunkSize, self.dataset.chunkSize),
            },
            "optimizer": self.optimizer.state_dict(),
            "model_states": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

        torch.save(checkpoint, self.model_output_location)
    
