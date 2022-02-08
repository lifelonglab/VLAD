from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    name: str
    data: np.array
    labels: np.array
