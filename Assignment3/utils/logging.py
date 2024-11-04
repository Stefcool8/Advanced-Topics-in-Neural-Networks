from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_epoch_metrics(self, epoch, model, train_loader, test_loader):
        device = next(model.parameters()).device  # Get the device of the model

        # Evaluate training loss
        train_loss = self.evaluate_loss(model, train_loader, device)

        # Evaluate test loss
        test_loss = self.evaluate_loss(model, test_loader, device)

        # Log the losses
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/test", test_loss, epoch)

    @staticmethod
    def evaluate_loss(model, data_loader, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)  # Transfer to GPU
                output = model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target).item()
        return total_loss / len(data_loader)


def get_logger(config):
    if config["logging"].get("tensorboard", False):
        return TensorBoardLogger()
    else:
        raise ValueError(f"Unsupported logging method: {config['logging']}")
