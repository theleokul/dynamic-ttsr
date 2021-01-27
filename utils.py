import typing as T

import yaml
import torch.nn as nn


def parse_config(config: str) -> T.Dict:
    with open(config, 'r') as f:
        parsed_config = yaml.safe_load(f)

    return parse_config


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
