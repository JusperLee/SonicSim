import os
import typing as T
import torch
import numpy as np
from SonicSim_rir import Scene
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rich import print

def save_xy_grid_points(room: str,
                        grid_distance: float,
                        dirname: str
                        ) -> T.Dict[str, T.Any]:
    """
    Save xy grid points given a mp3d room
    """

    filename_npy = f'{dirname}/grid_{room}.npy'
    filename_png = f'{dirname}/grid_{room}.png'

    scene = Scene(
        room,
        [None],  # placeholder for source class
        include_visual_sensor=False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    grid_points = scene.generate_xy_grid_points(grid_distance, filename_png=filename_png)
    room_size = scene.sim.pathfinder.navigable_area

    grid_info = dict(
        grid_points=grid_points,
        room_size=room_size,
        grid_distance=grid_distance,
    )
    np.save(filename_npy, grid_info)

    return grid_info

def load_room_grid(
    room: str,
    grid_distance: float
) -> T.Dict:
    """
    Load grid data for a specified room. If the grid data does not exist, it generates one.

    Args:
    - room:             Name of the room.
    - grid_distance:    The spacing between grid points.

    Returns:
    - A dictionary containing grid information for the specified room.
    """

    grid_distance_str = str(grid_distance).replace(".", "_")
    dirname_grid = f'data/scene_datasets/metadata/mp3d/grid_{grid_distance_str}'
    filename_grid = f'{dirname_grid}/grid_{room}.npy'
    if not os.path.exists(filename_grid):
        os.makedirs(dirname_grid, exist_ok=True)
        print(f'Computing grid_{room}...')
        grid_info = save_xy_grid_points(room, grid_distance, dirname_grid)

    # load grid
    grid_info = np.load(filename_grid, allow_pickle=True).item()

    return grid_info