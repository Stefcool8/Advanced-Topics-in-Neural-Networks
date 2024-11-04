import timm
from models import PreActResNet18, MLP, LeNet


class ModelFactory:
    @staticmethod
    def get_model(model_name, num_classes):
        if model_name == "lenet":
            return LeNet(num_classes=num_classes)
        elif model_name == "mlp":
            return MLP(num_classes=num_classes)
        elif model_name == "resnet18_cifar10":
            return timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
        elif model_name == "preact_resnet18":
            return PreActResNet18(num_classes=num_classes)
        elif model_name in timm.list_models():
            return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
