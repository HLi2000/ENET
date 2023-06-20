import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
"""
Improved EczemaNet's prediction model
"""

from src.models.components.pyramidpooling import SpatialPyramidPooling

# from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

class Branch(torch.nn.Module):

    def __init__(self, in_channels, num_classes, dropout):
        super().__init__()

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_channels, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=dropout),
        #     torch.nn.Linear(512, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=dropout),
        #     torch.nn.Linear(512, num_classes)
        # )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_channels, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, num_classes)
        # )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_channels, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=dropout),
        #     torch.nn.Linear(512, num_classes)
        # )

        ## 2FC (-Dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(in_channels, num_classes),
        #     torch.nn.Dropout(p=dropout),
        # )

    def forward(self, x):
        output = self.classifier(x)

        return output

# Regular PyTorch Module
class EczemaNet_New(torch.nn.Module):
    def __init__(self, num_classes, dropout, ordinal, ratio=False):
        super().__init__()

        if ordinal is True:
            num_classes = num_classes - 1

        self.base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # self.spp = SpatialPyramidPooling([1], 'max')

        num_channels = 1280
        # num_channels = self.spp.get_output_size(num_channels)

        self.ratio = ratio
        if self.ratio is True:
            num_channels += 1

        self.branch0 = Branch(num_channels, num_classes, dropout)
        self.branch1 = Branch(num_channels, num_classes, dropout)
        self.branch2 = Branch(num_channels, num_classes, dropout)
        self.branch3 = Branch(num_channels, num_classes, dropout)
        self.branch4 = Branch(num_channels, num_classes, dropout)
        self.branch5 = Branch(num_channels, num_classes, dropout)
        self.branch6 = Branch(num_channels, num_classes, dropout)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        ## Skin/Whole ratio
        if self.ratio is True:
            ratio = torch.sum(torch.sum(x, 1) > 0, dim=(1, 2)) / (x.shape[2]*x.shape[3])
            ratio = torch.unsqueeze(ratio, 1)
            # print(ratio)
            # print(ratio.shape)

        x = self.base.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        # print(x.size())
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # x = self.spp(x)
        # print(x.size())

        ## add Skin/Whole ratio
        if self.ratio is True:
            x = torch.concat([x, ratio], 1)
            # print(x[0])
            # print(x.shape)

        x0 = self.activation(self.branch0(x))
        x1 = self.activation(self.branch1(x))
        x2 = self.activation(self.branch2(x))
        x3 = self.activation(self.branch3(x))
        x4 = self.activation(self.branch4(x))
        x5 = self.activation(self.branch5(x))
        x6 = self.activation(self.branch6(x))
        return torch.stack([x0, x1, x2, x3, x4, x5, x6], axis=1)
