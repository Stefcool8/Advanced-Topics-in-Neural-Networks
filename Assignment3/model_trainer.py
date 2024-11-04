import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset_config.dataset_factory import DatasetFactory
from models.model_factory import ModelFactory
from utils.logging import get_logger
from utils.early_stopping import EarlyStopping
from torch import GradScaler, autocast
from tqdm import tqdm
import wandb


class Trainer:
    def __init__(self, config, sweep_config=None):
        self.config = config
        self.device = config["training"]["device"]
        if not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"
            print("CUDA not available, switching to CPU")
        self.device = torch.device(self.device)

        if sweep_config:
            self.model = ModelFactory.get_model(sweep_config.model, 100).to(self.device)
            if sweep_config.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=sweep_config.learning_rate)
            elif sweep_config.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=sweep_config.learning_rate, momentum=0.9,
                                                 weight_decay=0.0005, nesterov=True)
        else:
            self.model = ModelFactory.get_model(config["model"]["name"], config["model"]["num_classes"]).to(self.device)
            self.optimizer = getattr(torch.optim, config["training"]["optimizer"])(self.model.parameters(),
                                                                                   **config["training"][
                                                                                       "optimizer_args"])

        self.scheduler = getattr(torch.optim.lr_scheduler, config["training"]["scheduler"])(self.optimizer,
                                                                                            **config["training"][
                                                                                                "scheduler_args"])

        train_dataset = DatasetFactory.get_dataset(config, train=True,
                                                   dataset_transforms=sweep_config.transforms if sweep_config else
                                                   config["dataset"]["transforms"])
        test_dataset = DatasetFactory.get_dataset(config, train=False)

        self.train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True,
                                       pin_memory=True)
        self.val_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False,
                                     pin_memory=True)

        self.logger = get_logger(config)
        self.enable_half = self.device == 'cuda'
        self.scaler = GradScaler(self.device, enabled=self.enable_half)

        self.early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"],
                                            min_delta=config["training"]["early_stopping"]["min_delta"]) \
            if config["training"]["early_stopping"].get("enabled", False) else None

        cutmix = v2.CutMix(num_classes=100, alpha=1.0)
        mixup = v2.MixUp(num_classes=100, alpha=0.4)
        self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def train(self):
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        self.model.train()
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            # Apply CutMix to the batch
            inputs, targets = self.cutmix_or_mixup(inputs, targets)

            with autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(dim=1)).sum().item()

        return 100.0 * correct / total

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    @torch.inference_mode()
    def inference(self):
        self.model.eval()
        labels = []
        for inputs, _ in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            with autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)
            predicted = outputs.argmax(1).tolist()
            labels.extend(predicted)
        return labels

    def train_loop(self):
        best = 0.0
        epochs = list(range(self.config["training"]["epochs"]))
        with tqdm(epochs) as tbar:
            for epoch in tbar:
                train_acc = self.train()
                val_acc = self.val()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

                if val_acc > best:
                    best = val_acc
                tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")
                wandb.log({"train_accuracy": train_acc, "epoch": epoch})
                wandb.log({"validation_accuracy": val_acc, "epoch": epoch})

                self.logger.log_epoch_metrics(epoch, self.model, self.train_loader, self.val_loader)

                if self.early_stopping:
                    self.early_stopping(val_acc)
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered!")
                        break
