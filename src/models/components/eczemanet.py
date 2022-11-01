import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class Branch(torch.nn.Module):

    def __init__(self, in_channels, num_classes, dropout):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        output = self.classifier(x)

        return output

# Regular PyTorch Module
class EczemaNet(torch.nn.Module):
    def __init__(self, num_classes, dropout, ordinal):
        super().__init__()

        if ordinal is True:
            num_classes = num_classes - 1

        self.base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        self.branch0 = Branch(1280, num_classes, dropout)
        self.branch1 = Branch(1280, num_classes, dropout)
        self.branch2 = Branch(1280, num_classes, dropout)
        self.branch3 = Branch(1280, num_classes, dropout)
        self.branch4 = Branch(1280, num_classes, dropout)
        self.branch5 = Branch(1280, num_classes, dropout)
        self.branch6 = Branch(1280, num_classes, dropout)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.base.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x0 = self.activation(self.branch0(x))
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3(x))
        x4 = self.activation(self.branch4(x))
        x5 = self.activation(self.branch5(x))
        x6 = self.activation(self.branch6(x))
        return torch.stack([x0, x1, x2, x3, x4, x5, x6], axis=1)
