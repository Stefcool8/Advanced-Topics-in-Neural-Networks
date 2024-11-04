import wandb
from model_trainer import Trainer
from utils.config_loader import load_config

def load_sweep_config():
    return {
        'method': 'grid',
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'model': {
                'values': ['resnet18_cifar10', 'preact_resnet18']
            },
            'transforms': {
                'values': [['RandomHorizontalFlip', 'RandomRotation', 'ColorJitter'], 
                           ['RandomErasing', 'RandomPerspective', 'RandomCrop', 'GaussianBlur']]
            },
            'optimizer': {
                'values': ['sgd', 'adam']
            },
            'learning_rate': {
                'values': [0.01, 0.1]
            }
        }
    }


def main():
    run = wandb.init()
    trainer = Trainer(config, run.config)
    trainer.train_loop()


wandb.login()
sweep_config = load_sweep_config()

config = load_config('/kaggle/input/config/config.yaml')
# config = load_config('config.yaml')

sweep_id = wandb.sweep(sweep=sweep_config, project="Assignment3")

wandb.agent(sweep_id, function=main, count=8)

wandb.finish()
