# Note: add option to specifiy seperate testing set
# models where testing and training data are from the same set can be overfit and not generalize well

import torch
from torch.utils.data import DataLoader, random_split
from ..model.QLearnUNet import QUNet
from .QLearnDataset import QDataset, QDataLoader
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
    def __init__(self, dataset: QDataset, output_loc: str, feedback: QgsProcessingFeedback, args: dict, checkpoint: dict):

        # Set Training Arguments
        self.dataset = dataset                                          # Dataset used for DataLoader
        self.device = args["DEVICE"]                                    # CPU or GPU
        self.task = args["TRAIN_TYPE"]                                  # regression or classification
        self.epochs = args["EPOCHS"]                                    # Number of epochs to train for
        self.learning_rate = args["LEARNING_RATE"]                      # Learning Rate
        self.NODATA = args["NODATA"]
        self.batch_size = args["BATCH_SIZE"]
        self.val_split = args["VALIDATION_SPLIT"]                       # 0-1 ratio of data used for validation vs data used for training
        self.model_output_location = output_loc                         # Where to save model file
        self.mbase_channels=args["M_CHANNELS"]                          # Base channels for UNet
        self.mdepth=args["M_DEPTH"]                                     # Depth of UNet
        self.class_weight=args["CLASS_WEIGHTS"]                         # Class weight for CrossEntropyLoss
        self.mretain_dim=True
        self.save_mode = args["SAVE_MODE"]                              # Save mode (0 = best model, 1 = last model)
        self.best_loss = float("inf")                                   # Best model loss
        self.end_patience = args["END_PATIENCE"]                        # Early stopping patience
        self.epochs_no_improvement = 0                                  # Number of epochs with no improvement
        self.feedback = feedback                                        # For processing algorithm
        self.NODATA_class_mapping = self.dataset.NODATA_class_mapping   
        self.checkpoint = checkpoint

        # number of classes is set to 1 if regression
        # number of classes is automatically determined from input dataset if classification
        self.n_classes = len(self.dataset.class_mapping) if self.task == "classification" else 1
                                                                        
        # Setup PyTorch Training Objects
        self.setup_model()
        self.setup_dataloaders(self.dataset.PyTorchDataset)
        self.setup_OSL()

        self.load_checkpoint_data()

        self.feedback.pushInfo(f"Number of Classes: {self.n_classes} detected.")


    # try loading states of current model and optimizers to continue training
    def load_checkpoint_data(self):
        if self.checkpoint is None: # No checkpoint data (new training)
            return

        # Load model state
        self.model.load_state_dict(self.checkpoint["model_states"])
        self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        self.scheduler.load_state_dict(self.checkpoint["scheduler"])


    # Setup the optimizer, scheduler, and loss function
    def setup_OSL(self) -> None:
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) 
        # Reduce Learning Rate on Plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau( 
            self.optimizer, mode='min', factor=0.1, patience=4,min_lr=1e-6)
        # CrossEntropyLosss if Classification, MSELoss otherwise
        class_weights = torch.tensor(self.class_weight) if self.class_weight is not None else None
        self.criterion = (nn.CrossEntropyLoss(weight=class_weights,ignore_index=self.NODATA_class_mapping) 
                if self.task == "classification" else nn.MSELoss())
        self.feedback.pushInfo(f"ignore Index: {self.NODATA_class_mapping}")

    # setup the UNet model parameters
    # will use parameters from checkpoint if avaliable
    def setup_model(self) -> None:

        # load model params from checkpoint if available
        if self.checkpoint is not None:
            model_params = self.checkpoint["model_params"]
            self.mbase_channels = model_params["base_channels"]
            self.mdepth = model_params["depth"]
            self.mretain_dim = model_params["retain_dim"]
            

        self.model: QUNet = QUNet(                                      # UNet Model Init
            in_channels=self.dataset.bands,                             # Number of bands in input image
            base_channels=self.mbase_channels,                          #
            depth=self.mdepth,                                          # Depth of UNET, higher depth = longer training but more complex pattern recognition
            num_class=self.n_classes,                                   # Number of classes to generate for output
            retain_dim=self.mretain_dim,                                #
            out_sz=(self.dataset.chunkSize, self.dataset.chunkSize)     # Chunk Size
        ).to(self.device)                                               # Set device to be used for processing

    # Configure the dataloaders for training and validation based on the validation split
    def setup_dataloaders(self, dataset: QDataLoader) -> None:

        # Calculate training & validation dataset split
        val_size = int(self.val_split * len(dataset))
        gen = torch.Generator().manual_seed(42) # for reproducible results
        self.train_dataset, self.val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=gen)

        # Set up training data loader
        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Set up validation data loader
        self.val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        assert len(self.train_dataset) > 0, "Training dataset is empty"
        assert len(self.val_dataset) > 0, "Validation dataset is empty"

    def evaluate_model(self):
        if not hasattr(self.dataset, "TestPyTorchDataset"):
            self.feedback.pushInfo("No test dataset found")
            return
        
        self.feedback.pushInfo("Evaluating Model...")
        # Set up test data loader
        test_dl = DataLoader(self.dataset.TestPyTorchDataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_valid = 0

        with torch.no_grad():
            for images, targets in test_dl:
                # training was cancelled -> exit
                if self.checkCancel():
                    return
                
                images, targets = images.to(self.device), targets.to(self.device)
                targets = self.prepare_targets(targets)
                
                # Calculate loss
                outputs = self.model(images)
                loss = self.criterion_loss(outputs, targets, images)
                total_loss += loss.item()
                if self.task == "classification":
                    correct, valid = self.calculate_pred_accuracy(outputs, targets)
                    total_correct += correct
                    total_valid += valid

        # Finalize the accuracy calculations
        metrics = TrainingMetrics(loss=total_loss / len(test_dl))
        
        self.feedback.pushInfo("---------- Model Evaluation ----------")
        if self.task == "classification":
            metrics.accuracy = total_correct / total_valid if total_valid > 0 else 0.0
            self.feedback.pushInfo(f"Test Accuracy: {metrics.accuracy:.2%}")
        self.feedback.pushInfo(f"Test Loss: {metrics.loss:.4f}")
        self.feedback.pushInfo("Model Evaluation Finished!")
        self.feedback.pushInfo("------------------------------------------")


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
            loss = self.criterion_loss(outputs, targets, images)

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
    
    def criterion_loss(self, outputs: torch.tensor, targets: torch.tensor, inputs: torch.tensor) -> float:

        # mask NODATA values for regression -> MSELoss does not have ignore_index so it's very important to remove NODATA
        if self.task == "regression":
            mask = (targets != self.NODATA)
            outputs = outputs[mask] # remove nodata from outputs
            targets = targets[mask] # remove nodata from targets

        return self.criterion(outputs, targets)
    
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
                loss = self.criterion_loss(outputs, targets, images)

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
            val_metrics = None
            train_metrics = None

            # Catch interrupt raised by checkCancelled() and force stop training
            try:
                # Preform training and validation for one epoch
                train_metrics = self.train_epoch()
                val_metrics = self.val_epoch()
            except KeyboardInterrupt:
                return

            self.log_progress(epoch,train_metrics,val_metrics)
            self.scheduler.step(val_metrics.loss)

            # Check early stopping conditions
            if val_metrics.loss < self.best_loss:
                    self.best_loss = val_metrics.loss
                    self.epochs_no_improvement = 0

                    # Save the best model
                    if self.save_mode == 0:
                        self.feedback.pushInfo("Saving best model...")
                        self.save_model()
            else:
                self.epochs_no_improvement += 1
                if self.epochs_no_improvement >= self.end_patience:
                    self.feedback.pushInfo(f"Early stopping at epoch {epoch+1}...")
                    break
            

        self.feedback.pushInfo("Training Finished!")

        self.evaluate_model() # Evaluate the model on the test set (if available)

        # Save the last model if specified
        if self.save_mode == 1:
            self.save_model()
        self.dataset.clear_temp_dir()

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
            raise KeyboardInterrupt # Raise interrupt so we can catch in outer loop
        return False

    # Prepares target tensors for training
    def prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            targets = targets.long() # For CrossEntropyLoss
            # If 1 batch -> remove batch dimension
            if targets.ndim == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
        else:
            targets = targets.float() # For MSELoss
            if targets.ndim == 3:
                targets = targets.unsqueeze(1)

        return targets
        
    # compute correct and valid pixels
    def calculate_pred_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
        mask = targets != self.NODATA_class_mapping  # Mask NODATA values
        valid_pixels = mask.sum().item()
        
        # Early exit
        if valid_pixels == 0:
            return 0, 0
        
        preds = outputs.argmax(dim=1)
        # issue with calculating prediction accuracy -> probaby because preds have NODATA values in them
        correct = (preds[mask] == targets[mask]).sum().item()
        return correct, valid_pixels

    # Note: track loss and save best model
    # Note: preform training, validation, and testing 
    def save_model(self) -> None:
        # TODO: refactor into a dataclass with methods for properly loading and saving
        self.feedback.pushInfo(f"Saving model to {self.model_output_location}")
        checkpoint = {
            "model_params": {
                "in_channels": self.dataset.bands,
                "base_channels": self.mbase_channels,
                "depth": self.mdepth,
                "num_class": self.n_classes,
                "retain_dim": self.mretain_dim,
                "out_sz": (self.dataset.chunkSize, self.dataset.chunkSize)
            },
            "training_params": {
                "NODATA": self.NODATA,
                "NODATA_CLASS_MAPPING": self.NODATA_class_mapping,
                "task_type": self.task,
                "normalize_inputs": self.dataset.normalize_inputs,
                "normalize_targets": self.dataset.normalize_targets,
                "normalization_params_train": self.dataset.norm_params_train,
                "normalization_params_target": self.dataset.norm_params_target,
                "do_class_mapping": self.dataset.do_class_mapping,
                "class_mapping": self.dataset.class_mapping,
                "inv_class_mapping": self.dataset.inv_class_mapping
            },
            "optimizer": self.optimizer.state_dict(),
            "model_states": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

        torch.save(checkpoint, self.model_output_location)
    
