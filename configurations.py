# configurations.py

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrussConfig:
    # Material attributes
    area: float = 1.0
    emod: float = 10000.0
    density: float = 1.0

    # Domain attributes
    x: int = 0
    y: int = 0
    xm: int = 0
    yn: int = 0
    initial_state: List[List[int]] = field(default_factory=list)
    passive_nodes: List[int] = field(default_factory=list)
    max_volume: float = 0.0

    # Additional attributes
    node_sections: List[List[int]] = field(default_factory=list)
    optimal_displacement: Optional[float] = None
    optimal_states: List[List[List[int]]] = field(default_factory=list)
    max_states: Optional[int] = None
    optimal_strain_energy: Optional[float] = None

    # New attribute for displacement direction
    displacement_direction: str = "y"


@dataclass
class CantileverConfig(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 50, 20
        self.xm, self.yn = 3, 6
        self.initial_state = [
            [0, self.yn + 1],
            [self.yn + 1, 2 * self.yn],
            [0, self.yn],
            [self.yn, self.yn + 1],
            [self.yn, 2 * self.yn],
        ]
        self.passive_nodes = []
        self.max_volume = 1000.0
        self.displacement_direction = "y"
        self.construction_type = "progressive"
        self.max_element_length = 20


@dataclass
class LShapeConfig(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 40, 40
        self.xm, self.yn = 5, 5
        self.initial_state = [
            [2, 10],
            [2, 12],
            [2, 14],
            [10, 12],
            [10, 20],
            [10, 22],
            [12, 14],
            [12, 22],
            [20, 22],
        ]
        passive1 = range(3 * self.xm + 3, 4 * self.xm)
        passlist1 = list(passive1)
        passive2 = range(4 * self.xm + 3, 5 * self.xm)
        passlist2 = list(passive2)
        self.passive_nodes = passlist1 + passlist2
        self.max_volume = 500.0  # Example value
        self.displacement_direction = "y"
        self.construction_type = "progressive"
        self.max_element_length = 20


@dataclass
class Ororbia1Config(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 20, 30
        self.xm, self.yn = 4, 3
        self.initial_state = [[0, 11], [2, 11]]
        self.passive_nodes = []
        self.max_volume = 160.0
        self.optimal_displacement = 0.0895
        self.optimal_strain_energy = 0.4476
        self.optimal_states = [
            [[0, 7], [0, 9], [2, 7], [2, 11], [7, 9], [7, 11], [9, 11]]
        ]
        self.max_states = 2
        self.displacement_direction = "x"
        self.construction_type = "static"
        self.max_element_length = None


@dataclass
class Ororbia2Config(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 20, 40
        self.xm, self.yn = 5, 3
        self.initial_state = [[0, 8], [2, 8], [0, 14], [8, 14]]
        self.passive_nodes = []
        self.max_volume = 240.0
        self.optimal_displacement = 0.1859
        self.optimal_strain_energy = 1.4658
        self.optimal_states = [
            [
                [2, 8],
                [8, 14],
                [10, 0],
                [10, 14],
                [10, 8],
                [7, 0],
                [7, 8],
                [7, 2],
                [7, 10],
                [12, 10],
                [12, 0],
                [12, 14],
            ]
        ]
        self.max_states = 3
        self.displacement_direction = "x"
        self.construction_type = "static"
        self.max_element_length = None


@dataclass
class Ororbia3Config(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 40, 40
        self.xm, self.yn = 5, 5
        self.initial_state = [[0, 24], [4, 24]]
        self.passive_nodes = []
        self.max_volume = 225.0
        self.optimal_displacement = 0.0361
        self.optimal_strain_energy = 0.2301
        self.optimal_states = [
            [
                [0, 7],
                [0, 12],
                [0, 16],
                [4, 7],
                [7, 12],
                [7, 24],
                [12, 16],
                [12, 24],
                [16, 24],
            ]
        ]
        self.max_states = 3
        self.displacement_direction = "x_and_y"
        self.construction_type = "static"
        self.max_element_length = None


@dataclass
class Ororbia4Config(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 80, 40
        self.xm, self.yn = 5, 9
        self.initial_state = [[9, 26], [27, 26]]
        self.passive_nodes = []
        self.max_volume = 305.0
        self.optimal_displacement = 0.5916
        self.optimal_strain_energy = 0.0
        self.optimal_states = [
            [
                [30, 27],
                [30, 26],
                [30, 9],
                [0, 9],
                [0, 30],
                [6, 0],
                [6, 26],
                [6, 30],
            ],
            [
                [30, 27],
                [30, 26],
                [30, 9],
                [6, 9],
                [6, 26],
                [6, 30],
            ],
            [[30, 27], [30, 26], [30, 9], [9, 26]],
        ]
        self.max_states = 3
        self.node_sections = []
        self.displacement_direction = "x_and_y"
        self.construction_type = "static"
        self.max_element_length = None


@dataclass
class Ororbia5Config(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 80, 40
        self.xm, self.yn = 5, 5
        self.initial_state = [
            [0, 2],
            [2, 4],
            [2, 22],
            [0, 22],
            [4, 22],
        ]
        self.passive_nodes = []
        self.max_volume = 480.0
        self.optimal_displacement = 0.0390
        self.optimal_strain_energy = 0.0
        self.optimal_states = [
            [
                [0, 2],
                [0, 16],
                [0, 21],
                [2, 4],
                [2, 16],
                [2, 18],
                [2, 22],
                [4, 18],
                [4, 23],
                [16, 21],
                [16, 22],
                [18, 22],
                [18, 23],
                [21, 22],
                [22, 23],
            ]
        ]
        self.max_states = 4
        self.displacement_direction = "y"
        self.construction_type = "static"
        self.max_element_length = None





@dataclass
class Ororbia7Config(TrussConfig):
    def __post_init__(self):
        self.x, self.y = 60, 60
        self.xm, self.yn = 7, 7
        self.initial_state = [[0, 48], [6, 48]]
        self.passive_nodes = []
        self.max_volume = 350.0
        self.optimal_displacement = 0.0420
        self.optimal_strain_energy = 0.0
        self.optimal_states = [
            [
                [0, 8],
                [0, 17],
                [0, 23],
                [6, 17],
                [8, 17],
                [17, 23],
                [17, 24],
                [17, 48],
                [23, 24],
                [23, 48],
                [24, 48],
            ]
        ]
        self.max_states = 4
        self.displacement_direction = "x_and_y"
        self.construction_type = "static"
        self.max_element_length = None


# Collect configurations in a dictionary
CONFIGURATIONS = {
    'cantilever': CantileverConfig(),
    'Lshape': LShapeConfig(),
    'Ororbia_1': Ororbia1Config(),
    'Ororbia_2': Ororbia2Config(),
    'Ororbia_3': Ororbia3Config(),
    'Ororbia_4': Ororbia4Config(),
    'Ororbia_5': Ororbia5Config(),
    'Ororbia_7': Ororbia7Config(),
    # Add other configurations as needed...
}
