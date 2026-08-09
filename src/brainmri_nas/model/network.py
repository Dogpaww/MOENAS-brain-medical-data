"""Network builder (handoff §14).

`build_model(...)` is the single, explicit entry point for turning a decoded
`NetworkGenotype` into a fresh `nn.Module`. No global dataset state, no
random decoding, no hardcoded classifier dimension (`AdaptiveAvgPool2d(1)`
feeds a `Linear(c_prev, num_classes)`, so the flatten size always matches
whatever channel count the cells actually produced), and no auxiliary head
(handoff §30 item 9 calls out an unused auxiliary loss as a bug -- simplest
fix is to not build one at all until something needs it).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from brainmri_nas.model.cell import Cell
from brainmri_nas.search_space.genotype import NetworkGenotype

SUPPORTED_STEM_TYPES = ("cifar",)


class NASNetwork(nn.Module):
    def __init__(
        self,
        genotype: NetworkGenotype,
        *,
        input_channels: int,
        num_classes: int,
        image_size: int,
        initial_channels: int,
        number_of_cells: int,
        drop_path_probability: float,
        stem_type: str,
        classifier_dropout_probability: float = 0.0,
    ):
        super().__init__()
        if stem_type not in SUPPORTED_STEM_TYPES:
            raise NotImplementedError(
                f"stem_type={stem_type!r} is not implemented yet. Supported: {SUPPORTED_STEM_TYPES}. "
                "Larger input resolutions need their own stem (handoff §5/§15), not the CIFAR-style one."
            )
        if number_of_cells < 3:
            raise ValueError(
                f"number_of_cells={number_of_cells} is too small for the reduction-cell schedule "
                "(indices layers//3 and 2*layers//3 would coincide or cover the whole network)."
            )

        self.genotype = genotype
        self.image_size = image_size
        self.drop_path_probability = drop_path_probability

        stem_multiplier = 3
        c_curr = stem_multiplier * initial_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, initial_channels
        reduction_idx = {number_of_cells // 3, 2 * number_of_cells // 3}

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(number_of_cells):
            reduction = i in reduction_idx
            if reduction:
                c_curr *= 2

            cell_genotype = genotype.reduction if reduction else genotype.normal
            cell = Cell(
                cell_genotype,
                c_prev_prev,
                c_prev,
                c_curr,
                reduction,
                reduction_prev,
                drop_path_probability=drop_path_probability,
            )
            self.cells.append(cell)

            reduction_prev = reduction
            c_prev_prev, c_prev = c_prev, cell.multiplier * c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(p=classifier_dropout_probability)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        out = self.classifier_dropout(out)
        return self.classifier(out)


def build_model(
    genotype: NetworkGenotype,
    *,
    input_channels: int,
    num_classes: int,
    image_size: int,
    initial_channels: int,
    number_of_cells: int,
    drop_path_probability: float,
    stem_type: str,
    classifier_dropout_probability: float = 0.0,
) -> nn.Module:
    return NASNetwork(
        genotype,
        input_channels=input_channels,
        num_classes=num_classes,
        image_size=image_size,
        initial_channels=initial_channels,
        number_of_cells=number_of_cells,
        drop_path_probability=drop_path_probability,
        stem_type=stem_type,
        classifier_dropout_probability=classifier_dropout_probability,
    )
