import os
import sys
import math
import random
import magnum as mn
import typing as T
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from contextlib import contextmanager
from rich import print
import numpy as np
import torch
import torchaudio
import cv2
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple
import habitat_sim.sim
from habitat_sim.utils import quat_from_angle_axis
from habitat.utils.visualizations import maps, utils

from tqdm import tqdm
import torch.multiprocessing as mp

def clip_all(audio_list):
    """
    Clips all audio signals in a list to the same length.

    Args: 
        audio_list: List of audio signals.

    Returns: 
    - List of audio signals of the same length.
    """

    min_length = min(audio.shape[-1] for audio in audio_list)
    clipped_audio_list = []
    for audio in audio_list:
        clipped_audio = audio[..., :min_length]
        clipped_audio_list.append(clipped_audio)

    return clipped_audio_list

@contextmanager
def suppress_stdout_and_stderr():
    """
    To suppress the logs from SoundSpaces
    """

    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        os.dup2(original_stderr_fd, sys.stderr.fileno())
        os.close(devnull)

def fft_conv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    is_cpu: bool = False
) -> torch.Tensor:
    """
    Perform convolution of a signal and a kernel using Fast Fourier Transform (FFT).

    Args:
    - signal (torch.Tensor): Input signal tensor.
    - kernel (torch.Tensor): Kernel tensor.
    - is_cpu (bool, optional): Flag to determine if the operation should be on the CPU.

    Returns:
    - torch.Tensor: Convolved signal.
    """

    if is_cpu:
        signal = signal.detach().cpu()
        kernel = kernel.detach().cpu()

    padded_signal = F.pad(signal.reshape(-1), (0, kernel.size(-1) - 1))
    padded_kernel = F.pad(kernel.reshape(-1), (0, signal.size(-1) - 1))

    signal_fr = torch.fft.rfftn(padded_signal, dim=-1)
    kernel_fr = torch.fft.rfftn(padded_kernel, dim=-1)

    output_fr = signal_fr * kernel_fr
    output = torch.fft.irfftn(output_fr, dim=-1)

    return output

class Receiver:
    """
    Receiver for SoundSpaces
    """

    def __init__(self,
                 position: T.Tuple[float, float, float],
                 rotation: float,
                 sample_rate: float = 16000,
                 ):

        self.position = position
        self.rotation = rotation
        self.sample_rate = sample_rate


class Source:
    """
    Source for Soundspaces
    """

    def __init__(self,
                 position: T.Tuple[float, float, float],
                 rotation: float,
                 dry_sound: str,
                 device: torch.device
                 ):

        self.position = position
        self.rotation = rotation
        self.device = device  # where to store dry_sound
        self.dry_sound = dry_sound


class Scene:
    """
    Soundspaces scene including room, receiver, and source list
    """

    def __init__(self,
                 room: str,
                 source_name_list: T.List[str],
                 receiver: Receiver = None,
                 source_list: T.List[Source] = None,
                 include_visual_sensor: bool = True,
                 device: torch.device = torch.device('cpu'),
                 image_size: T.Tuple[int, int] = (512, 256),
                 hfov: float = 90.0,
                 use_default_material: bool = False,
                 channel_type: str = 'Ambisonics',
                 channel_order: int = 1
                 ):

        # Set scene
        self.room = room
        self.n_sources = len(source_name_list)
        assert self.n_sources > 0
        self.receiver = receiver
        self.source_list = source_list
        self.source_current = None
        self.include_visual_sensor = include_visual_sensor
        self.device = device  # where to store IR

        # Set channel config for soundspaces
        self.channel = {}
        self.channel['type'] = channel_type
        self.channel['order'] = channel_order
        if channel_type == 'Ambisonics':
            self.channel_count = (self.channel['order'] + 1)**2
        elif channel_type == 'Binaural':
            self.channel_count = 2
        elif channel_type == 'Mono':
            self.channel_count = 1

        # Set aihabitat config for soundspaces
        self.aihabitat = {}
        self.aihabitat['default_agent'] = 0
        self.aihabitat['sensor_height'] = 1.5
        self.aihabitat['height'] = image_size[0]
        self.aihabitat['width'] = image_size[1]
        self.aihabitat['hfov'] = hfov # 视觉传感器所使用的视场范围。

        # Set acoustics config for soundspaces
        self.acoustic_config = {}
        self.acoustic_config['sampleRate'] = 16000
        self.acoustic_config['direct'] = True
        self.acoustic_config['indirect'] = True
        self.acoustic_config['diffraction'] = True
        self.acoustic_config['transmission'] = True
        self.acoustic_config['directSHOrder'] = 5
        self.acoustic_config['indirectSHOrder'] = 3
        self.acoustic_config['unitScale'] = 1
        self.acoustic_config['frequencyBands'] = 32
        self.acoustic_config['indirectRayCount'] = 50000

        # Set audio material
        if use_default_material:
            self.audio_material = 'SonicSet/material/mp3d_material_config_default.json'
        else:
            self.audio_material = 'SonicSet/material/mp3d_material_config.json'

        # Create simulation
        self.create_scene()

        # Randomly set source and receiver position
        source_position, source_rotation = None, None
        receiver_position, receiver_rotation = None, None

        # Create receiver (inside the room)
        if self.receiver is None:
            # random receiver
            self.create_receiver(receiver_position, receiver_rotation)
        else:
            # input receiver
            self.update_receiver(self.receiver)

    def create_scene(self):
        """
        Given the configuration, create a scene for soundspaces
        """

        # Set backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = f'mp3d/{self.room}/{self.room}.glb'
        backend_cfg.scene_dataset_config_file = 'SonicSet/material/mp3d.scene_dataset_config.json'
        backend_cfg.load_semantic_mesh = True
        backend_cfg.enable_physics = False

        # Set agent configuration
        agent_config = habitat_sim.AgentConfiguration()

        if self.include_visual_sensor:
            # Set color sensor
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color_sensor"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = [self.aihabitat['height'], self.aihabitat['width']]
            rgb_sensor_spec.position = [0.0, self.aihabitat["sensor_height"], 0.0]
            rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE # “针孔”（Pinhole）相机模型
            rgb_sensor_spec.hfov = self.aihabitat["hfov"]
            agent_config.sensor_specifications = [rgb_sensor_spec]

            # Set depth sensor
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.aihabitat["height"], self.aihabitat["width"]]
            depth_sensor_spec.position = [0.0, self.aihabitat["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            depth_sensor_spec.hfov = self.aihabitat["hfov"]
            agent_config.sensor_specifications.append(depth_sensor_spec)

            # # Set semantic sensor
            # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            # semantic_sensor_spec.uuid = "semantic_sensor"
            # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            # semantic_sensor_spec.resolution = [self.aihabitat["height"], self.aihabitat["width"]]
            # semantic_sensor_spec.position = [0.0, self.aihabitat["sensor_height"], 0.0]
            # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            # semantic_sensor_spec.hfov = self.aihabitat["hfov"]
            # agent_config.sensor_specifications.append(semantic_sensor_spec)

        # Set simulator configuration
        cfg = habitat_sim.Configuration(backend_cfg, [agent_config])

        # Set simulator
        sim = habitat_sim.Simulator(cfg)

        # set navmesh path for searching for navigatable points
        navmesh = f'mp3d/{self.room}/{self.room}.navmesh'
        sim.pathfinder.load_nav_mesh(navmesh)

        # seed for navmesh
        sim.seed(random.randint(0, 1024))

        # Set simulation
        self.sim = sim
        print('Scene created!')

        return self

    def add_audio_sensor(self):
        """
        Add audio sensor to the scene
        """

        # set audio sensor
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        audio_sensor_spec.uuid = "audio_sensor"
        audio_sensor_spec.enableMaterials = True  # make sure _semantic.ply file is in the scene folder
        audio_sensor_spec.channelLayout.type = getattr(habitat_sim.sensor.RLRAudioPropagationChannelLayoutType, self.channel['type'])
        audio_sensor_spec.channelLayout.channelCount = self.channel_count  # ambisonics

        # Set acoustic configuration
        audio_sensor_spec.acousticsConfig.sampleRate = self.acoustic_config['sampleRate']
        audio_sensor_spec.acousticsConfig.direct = self.acoustic_config['direct']
        audio_sensor_spec.acousticsConfig.indirect = self.acoustic_config['indirect']
        audio_sensor_spec.acousticsConfig.diffraction = self.acoustic_config['diffraction']
        audio_sensor_spec.acousticsConfig.transmission = self.acoustic_config['transmission']
        audio_sensor_spec.acousticsConfig.directSHOrder = self.acoustic_config['directSHOrder']
        audio_sensor_spec.acousticsConfig.indirectSHOrder = self.acoustic_config['indirectSHOrder']
        audio_sensor_spec.acousticsConfig.unitScale = self.acoustic_config['unitScale']
        audio_sensor_spec.acousticsConfig.frequencyBands = self.acoustic_config['frequencyBands']
        audio_sensor_spec.acousticsConfig.indirectRayCount = self.acoustic_config['indirectRayCount']
        # audio_sensor_spec.acousticsConfig.maxIRLength = 40.0
        # audio_sensor_spec.acousticsConfig.sourceRayCount = 2000
        # audio_sensor_spec.acousticsConfig.meshSimplification = False

        # Initialize receiver
        audio_sensor_spec.position = [0.0, self.aihabitat['sensor_height'], 0.0]  # audio sensor has a height of 1.5m
        self.sim.add_sensor(audio_sensor_spec)

        audio_sensor = self.sim.get_agent(self.aihabitat['default_agent'])._sensors['audio_sensor']
        audio_sensor.setAudioMaterialsJSON(self.audio_material)

        return self

    def create_receiver(self,
                        position: T.Tuple[float, float, float] = None,
                        rotation: float = None
                        ):
        """
        Randomly sample receiver position and rotation
        """

        if position is None:
            # Randomly set receiver position in the room
            position = self.sim.pathfinder.get_random_navigable_point()
            rotation = random.uniform(0, 360)

        # Set sample rate
        sample_rate = self.acoustic_config['sampleRate']

        # Set receiver
        receiver = Receiver(position, rotation, sample_rate)

        # Update receiver
        self.update_receiver(receiver)

        return self

    def update_receiver(self,
                        receiver: Receiver
                        ):
        """
        Update receiver
        """

        agent = self.sim.get_agent(self.aihabitat["default_agent"])
        new_state = self.sim.get_agent(self.aihabitat["default_agent"]).get_state()
        new_state.position = np.array(receiver.position + np.array([0, 0.0, 0]))  # agent height is already applied in audio_sensor_spec.position
        new_state.rotation = quat_from_angle_axis(math.radians(receiver.rotation), np.array([0, 1.0, 0]))  # + -> left
        # new_state.rotation *= quat_from_angle_axis(math.radians(-30), np.array([1.0, 0, 0]))  # + -> up
        new_state.sensor_states = {}
        agent.set_state(new_state, True)

        self.receiver = receiver  # for reference

        return self

    def update_receiver_position(self,
                                 receiver_position: T.Tuple[float, float, float]
                                 ):
        """
        Update receiver position
        """

        self.receiver.position = receiver_position

        agent = self.sim.get_agent(self.aihabitat["default_agent"])
        new_state = self.sim.get_agent(self.aihabitat["default_agent"]).get_state()
        new_state.position = np.array(receiver_position + np.array([0, 0.0, 0]))  # agent height is already applied in audio_sensor_spec.position
        new_state.sensor_states = {}
        agent.set_state(new_state, True)

        return self

    def create_source(self,
                      source_name: str,
                      source_id: int,
                      position: T.Tuple[float, float, float] = None,
                      rotation: float = None
                      ):
        """
        Set source given the source name, position, and rotation
        """

        if position is None:
            # Randomly set source position in the room
            position = self.sim.pathfinder.get_random_navigable_point()
            rotation = random.uniform(0, 360)  # only for mesh as source sound is omnidirectional

        # Randomly set source sound
        dry_sound = ""

        # Set source
        source = Source(position, rotation, dry_sound, device=self.device)

        # Save source
        self.update_source(source, source_id)

        return self

    def update_source(self,
                      source: Source,
                      source_id: int = None
                      ):
        """
        Update source
        """

        if source_id is not None:
            # update source list
            self.source_list[source_id] = source
        else:
            # update current source
            audio_sensor = self.sim.get_agent(self.aihabitat['default_agent'])._sensors['audio_sensor']
            audio_sensor.setAudioSourceTransform(source.position + np.array([0, self.aihabitat["sensor_height"], 0]))  # add 1.5m to the height calculation

            self.source_current = source  # for reference

        return self

    def update_source_position(self,
                               source_position
                               ):
        """
        Update Source position
        """

        audio_sensor = self.sim.get_agent(self.aihabitat['default_agent'])._sensors['audio_sensor']
        audio_sensor.setAudioSourceTransform(source_position + np.array([0, self.aihabitat["sensor_height"], 0]))  # add 1.5m to the height calculation

    def render_ir(self,
                  source_id: int
                  ) -> torch.Tensor:
        """
        Render IR given the source ID
        """

        source = self.source_list[source_id]
        self.update_source(source)
        ir = torch.tensor(self.sim.get_sensor_observations()['audio_sensor'], device=self.device)

        return ir

    def render_ir_simple(self,
                         source_position: T.Tuple[float, float, float],
                         receiver_position: T.Tuple[float, float, float],
                         ) -> torch.Tensor:
        """
        Render IR given the source ID
        """

        # source
        self.update_source_position(source_position)

        # receiver
        self.update_receiver_position(receiver_position)

        # render ir
        ir = torch.tensor(self.sim.get_sensor_observations()['audio_sensor'], device=self.device)

        return ir

    def render_ir_all(self) -> T.List[torch.Tensor]:
        """
        Render IR for all sources
        """

        ir_list = []
        for source_id in range(self.n_sources):
            print(f'Rendering IR {source_id}/{self.n_sources}...')
            ir = self.render_ir(source_id)
            ir_list.append(ir)

        return ir_list

    def render_image(self,
                     is_instance=False
                     ):
        """
        Render image including rgb, depth, and semantic
        """

        observation = self.sim.get_sensor_observations()
        rgb = observation["color_sensor"]
        depth = observation["depth_sensor"]
        return rgb, depth

    def render_envmap(self):
        """
        Render environment map in *** format
        """

        with suppress_stdout_and_stderr():
            angles = [0, 270, 180, 90]
            rgb_panorama = []
            depth_panorama = []

            for angle_offset in angles:
                angle = self.receiver.rotation + angle_offset
                agent = self.sim.get_agent(self.aihabitat["default_agent"])
                new_state = self.sim.get_agent(self.aihabitat["default_agent"]).get_state()
                new_state.rotation = quat_from_angle_axis(
                    math.radians(angle), np.array([0, 1.0, 0])
                ) * quat_from_angle_axis(math.radians(0), np.array([1.0, 0, 0]))
                new_state.sensor_states = {}
                agent.set_state(new_state, True)

                observation = self.sim.get_sensor_observations()
                rgb_panorama.append(observation["color_sensor"])
                depth_panorama.append((observation['depth_sensor']))
            envmap_rgb = np.concatenate(rgb_panorama, axis=1)
            envmap_depth = np.concatenate(depth_panorama, axis=1)

            # rotate receiver to original angle
            self.update_receiver(self.receiver)

        return envmap_rgb, envmap_depth

    def generate_xy_grid_points(self,
                                grid_distance: float,
                                height: float = None,
                                filename_png: str = None,
                                meters_per_pixel: float = 0.005
                                ) -> torch.Tensor:
        """
        Generate the 3D positions of grid points at the given height
        """

        pathfinder = self.sim.pathfinder
        assert pathfinder.is_loaded
        # agent_height = pathfinder.nav_mesh_settings.agent_height  # to be navigable, full body of the agent should be inside
        if height is None:  # height of the agent foot
            height = 0
            # height = pathfinder.get_bounds()[0][1]  # floor height

        # Sample grid
        bounds = pathfinder.get_bounds()
        x_points = torch.arange(bounds[0][0], bounds[1][0] + grid_distance, grid_distance)
        z_points = torch.arange(bounds[0][2], bounds[1][2] + grid_distance, grid_distance)
        x_grid, z_grid = torch.meshgrid(x_points, z_points)
        y_value = height * torch.ones_like(x_grid.reshape(-1))

        # Combine x, y, and z coordinates into a single tensor of points
        points = torch.stack([x_grid.reshape(-1), y_value.reshape(-1), z_grid.reshape(-1)], dim=-1)
        grid_points = []
        for point in points:
            snapped_point = pathfinder.snap_point(point)
            if not np.any(np.isnan(snapped_point)):
                # 检查距离
                close_point = False
                for existing_point in grid_points:
                    distance = torch.norm(existing_point - torch.from_numpy(snapped_point))
                    if distance < grid_distance:
                        close_point = True
                        break
                if not close_point:
                    grid_points.append(torch.from_numpy(snapped_point))
        # torch.tensor(is_points_navigable).sum()
        
        # Flatten the tensor of points into a list
        grid_points = torch.stack(grid_points)

        # assert len(grid_points) > 0
        # save image
        if filename_png is not None:
            save_town_map_grid(filename_png, pathfinder, grid_points, meters_per_pixel=meters_per_pixel)

        return grid_points

    def generate_data(self, use_dry_sound: bool = False):
        """
        Generate all data including IR, envmap, audio, image
        """

        # env map
        if self.include_visual_sensor:
            envmap_rgb, envmap_depth = self.render_image()
        else:
            envmap_rgb, envmap_depth = None, None

        # IR
        self.add_audio_sensor()  # add audio_sensor after image rendering for faster image rendering
        ir_list = self.render_ir_all()
        # ir_total = sum_arrays_with_different_length(ir_list).detach().cpu()

        # audio_list
        dry_sound_list = []
        audio_list = []
        # audio_total = None
        if use_dry_sound:
            for source_id, source in enumerate(self.source_list):
                # load dry sound
                dry_sound = source.dry_sound
                if isinstance(dry_sound, str):
                    dry_sound, sample_rate = torchaudio.load(dry_sound)
                    self.dry_sound = dry_sound.to(self.device)
                    self.sample_rate = sample_rate

                ir = ir_list[source_id]
                audio = torch.stack([fft_conv(dry_sound[0], ir_channel, is_cpu=True) for ir_channel in ir])
                dry_sound_list.append(dry_sound.detach().cpu())
                audio_list.append(audio.detach().cpu())

        # cpu
        ir_list = [tensor.detach().cpu() for tensor in ir_list]

        return dict(
            ir_list=ir_list,
            sample_rate=self.receiver.sample_rate,
            envmap=[envmap_rgb, envmap_depth],
            audio_list=audio_list,
            dry_sound_list=dry_sound_list,
        )
        
def create_custom_arrayir(
    room: str,
    source_position: T.Tuple[float, float, float],
    receiver_position: T.Tuple[float, float, float],
    mic_array: T.List[T.Tuple[float, float, float]],
    filename: str = None,
    receiver_rotation: float = None,
    sample_rate: float = 16000,
    use_default_material: bool = False,
    channel_order: int = 1
):
    """
    Render impulse response for a source and receiver pair in the mp3d room.
    """

    if receiver_rotation is None:
        receiver_rotation = 90

    # Create a source
    source = Source(
        position=source_position,
        rotation=0,
        dry_sound='',
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    multi_channels = []
    for mic_idx, mic in enumerate(mic_array):
        # Create a receiver
        receiver = Receiver(
            position=receiver_position+mic,
            rotation=receiver_rotation,
            sample_rate=sample_rate
        )
        scene = Scene(
            room,
            [None],  # placeholder for source class
            receiver=receiver,
            source_list=[source],
            include_visual_sensor=False,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            use_default_material=use_default_material,
            channel_type="Mono",
            channel_order=channel_order
        )
        # Render IR
        scene.add_audio_sensor()
        ir = scene.render_ir(0)
        multi_channels.append(ir)
        scene.sim.close()
    # Save file if dirname is given
    multi_channels = clip_all(multi_channels) 
    multi_channels = torch.cat(multi_channels, dim=0)
    if filename is not None:
        torchaudio.save(filename, multi_channels, sample_rate=sample_rate)
    else:
        return multi_channels
        
def render_ir(room: str,
              source_position: T.Tuple[float, float, float],
              receiver_position: T.Tuple[float, float, float],
              filename: str = None,
              receiver_rotation: float = None,
              sample_rate: float = 16000,
              use_default_material: bool = False,
              channel_type: str = 'Ambisonics',
              channel_order: int = 1
              ) -> torch.Tensor:
    """
    Render impulse response for a source and receiver pair in the mp3d room.
    """

    if receiver_rotation is None:
        receiver_rotation = 90

    # Create a receiver
    receiver = Receiver(
        position=receiver_position,
        rotation=receiver_rotation,
        sample_rate=sample_rate
    )

    # Create a source
    source = Source(
        position=source_position,
        rotation=0,
        dry_sound='',
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )

    scene = Scene(
        room,
        [None],  # placeholder for source class
        receiver=receiver,
        source_list=[source],
        include_visual_sensor=False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        use_default_material=use_default_material,
        channel_type=channel_type,
        channel_order=channel_order
    )

    # Render IR
    scene.add_audio_sensor()
    # with suppress_stdout_and_stderr():
    ir = scene.render_ir(0).detach().cpu()
    scene.sim.close()
    # Save file if dirname is given
    if filename is not None:
        torchaudio.save(filename, ir, sample_rate=sample_rate)
    else:
        return ir


def render_rir_parallel(room_list, source_position_list, receiver_position_list,
                        mic_array_list=None, filename_list=None, receiver_rotation_list=None,
                        batch_size=64, sample_rate=16000, use_default_material=False,
                        channel_type='Ambisonics', channel_order=1):
    """
    Run render_ir (or create_custom_arrayir) in parallel for all elements.
    """
    assert len(room_list) == len(source_position_list)
    assert len(source_position_list) == len(receiver_position_list)

    if filename_list is None:
        is_return = True
    else:
        is_return = False

    if receiver_rotation_list is None:
        receiver_rotation_list = [0] * len(receiver_position_list)

    num_points = len(source_position_list)
    num_batches = (num_points + batch_size - 1) // batch_size

    progress_bar = tqdm(total=num_points)

    def update_progress(*_):
        progress_bar.update()

    # 使用 torch.multiprocessing.Manager()
    with mp.Manager() as manager:
        ir_list = manager.list() 
        with mp.Pool(mp.cpu_count()) as pool:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_points)

                if is_return:
                    batch = [
                        (room_list[i], source_position_list[i], receiver_position_list[i], None, receiver_rotation_list[i]) 
                        for i in range(start_idx, end_idx)
                    ]
                else:
                    batch = [
                        (room_list[i], source_position_list[i], receiver_position_list[i], filename_list[i], receiver_rotation_list[i])
                        for i in range(start_idx, end_idx)
                    ]

                tasks = []
                for room, source_position, receiver_position, filename, receiver_rotation in batch:
                    if channel_type == "CustomArrayIR":
                        task = pool.apply_async(create_custom_arrayir, 
                                                args=(room, source_position, receiver_position, mic_array_list, filename, receiver_rotation, sample_rate, use_default_material, channel_order),
                                                callback=update_progress)
                    else:
                        task = pool.apply_async(render_ir, 
                                                args=(room, source_position, receiver_position, filename, receiver_rotation, sample_rate, use_default_material, channel_type, channel_order),
                                                callback=update_progress)
                    tasks.append(task)

                for task in tasks:
                    if is_return:
                        ir = task.get()
                        # 确保返回 CPU 张量
                        ir_list.append(ir)
                    else:
                        task.get()

        if is_return:
            # 在返回前转换成普通列表，避免 manager 上下文关闭后再访问
            return list(ir_list)

def create_scene(room: str,
                 receiver_position: T.Tuple[float, float, float] = [0.0, 0.0, 0.0],
                 sample_rate: float = 16000,
                 image_size: T.Tuple[int, int] = (512, 256),
                 include_visual_sensor: bool = True,
                 hfov: float = 90.0
                 ) -> Scene:
    """
    Create a soundspaces scene to render IR.
    """

    # Note: Make sure mp3d room is downloaded
    with suppress_stdout_and_stderr():
        # Create a receiver
        receiver = Receiver(
            position=receiver_position,
            rotation=0,
            sample_rate=sample_rate
        )

        scene = Scene(
            room,
            [None],  # placeholder for source class
            receiver=receiver,
            include_visual_sensor=include_visual_sensor,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            image_size=image_size,
            hfov=hfov
        )

    return scene

def convert_points_to_topdown(
    pathfinder,
    points: T.List[T.Tuple[float, float, float]],
    meters_per_pixel: float
) -> T.List[T.Tuple[float, float]]:
    """
    Convert 3D points (x, z) to top-down view points (x, y).

    Args:
    - pathfinder:           The pathfinder object for conversion context.
    - points:               List of 3D points.
    - meters_per_pixel:     Scale for converting meters to pixels.

    Returns:
    - A list of converted points in top-down view.
    """

    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

# display a topdown map with matplotlib
def image_marker(image, xy, ax, zoom=0.05):
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(
        im,
        xy,
        xybox=(0.0, 0.0),
        xycoords="data",
        boxcoords="offset points",
        pad=0,
        frameon=False,
    )
    ax.add_artist(ab)

def display_map(
    topdown_map,
    filename: T.Optional[str] = None,
    key_points: T.Optional[T.List[T.Tuple[float, float]]] = None,
):
    """
    Display a top-down map. Optionally, plot key points on the map.

    Args:
    - topdown_map:      Topdown map
    - filename:         Filename to save the topdown map
    - key_points:       List of points to be highlighted on the map.
    - text_margin:      Margin for text labels, defaults to 5.
    - is_grid:          If True, considers the points as grid points.
    """
    # plot points on map
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # # 创建用于图例的空线条
    if key_points is not None:
        for i, point in enumerate(key_points):
            if i == 0:
                # receiver
                print(f"receiver: {point}")
                image = plt.imread('SonicSet/imgs/mic.png')
                image_marker(image, (point[0], point[1]), ax)
                # plt.plot(point[0], point[1], color='red', marker='o', markersize=10)

            if i in [1, 2, 3]:
                # speaker voice
                print(f"speaker {i}: {point}")
                image = plt.imread(f'SonicSet/imgs/{i}.png')
                image_marker(image, (point[0], point[1]), ax)
                # plt.plot(point[0], point[1], color='red', marker='o', markersize=10)
                
            if i in [4]:
                # noise
                print(f"noise: {point}")
                image = plt.imread(f'SonicSet/imgs/noise.png')
                image_marker(image, (point[0], point[1]), ax)

            if i in [5]:
                # music
                print(f"music: {point}")
                image = plt.imread(f'SonicSet/imgs/music.png')
                image_marker(image, (point[0], point[1]), ax)
    else:
        raise ValueError("key_points must be provided.")
    
    ax.plot([], [], color='red', label='Speaker 1')
    ax.plot([], [], color='blue', label='Speaker 2')
    ax.plot([], [], color='green', label='Speaker 3')
    ax.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show(block=False)

def save_town_map_grid(
    filename: str,
    pathfinder,
    grid_points: T.List[T.Tuple[float, float, float]],
    meters_per_pixel: float = 0.05,
    is_grid = True
):
    """
    Generate a top-down view of a town map with grid points

    Args:
    - filename:             Filename to save town map image
    - pathfinder:           Pathfinder object used for contextual conversion.
    - grid_points:          List of 3D grid points.
    - meters_per_pixel:     Scale for converting meters to pixels. Defaults to 0.05.
    """

    assert pathfinder.is_loaded
    grid_points = np.array(grid_points)

    if len(grid_points) == 0:
        height = 0  # for empty grid_points
        xy_grid_points = None
    else:
        height = grid_points[0, 1]
        # Convert points to topdown
        xy_grid_points = convert_points_to_topdown(
            pathfinder, grid_points, meters_per_pixel
        )

    # Get topdown map
    top_down_map = maps.get_topdown_map(
        pathfinder, height=height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]

    # Save map
    display_map(top_down_map, key_points=xy_grid_points, filename=filename, is_grid=is_grid)

def draw_path(
    top_down_map: np.ndarray,
    path_points: List[Tuple],
    color: Tuple[int, int, int] = (255, 0 ,0),
    thickness: int = 2,
) -> None:
    r"""Draw path on top_down_map (in place) with specified color.
    Args:
        top_down_map: A colored version of the map.
        color: color code of the path, from TOP_DOWN_MAP_COLORS.
        path_points: list of points that specify the path to be drawn
        thickness: thickness of the path.
    """
    for prev_pt, next_pt in zip(path_points[:-1], path_points[1:]):
        # Swapping x y
        cv2.line(
            top_down_map,
            prev_pt[::-1],
            next_pt[::-1],
            color,
            thickness=thickness,
        )

def save_trace_gif(
    scene: Scene,
    filename: str,
    grid_points: T.List[T.Tuple[float, float, float]],
    meters_per_pixel: float = 0.05
):
    """
    Generate a top-down view of a town map with trace
    """
    height = grid_points[0][0][0][1]
    # print(height)
    # Get topdown map
    top_down_map = maps.get_topdown_map(
        scene.sim.pathfinder, height=height, meters_per_pixel=meters_per_pixel, 
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    
    trajectorys = []
    colors = [
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0)
    ]
    for idx, spk_nav_points in enumerate(grid_points[0]):
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        # convert world trajectory points to maps module grid points
        trajectory = [
                maps.to_grid(
                    path_point[2],
                    path_point[0],
                    grid_dimensions,
                    pathfinder=scene.sim.pathfinder,
                )
                for path_point in spk_nav_points
            ]
        # draw the agent and trajectory on the map
        # import pdb; pdb.set_trace()
        draw_path(top_down_map, trajectory, color=colors[idx], thickness=1)
        trajectorys.append(trajectory)
    
    mic_xy_point = convert_points_to_topdown(scene.sim.pathfinder, [grid_points[1]], meters_per_pixel)
    mic_xy_point = [np.round(point).astype(int) for point in mic_xy_point]
    
    noise_music_xy_point = convert_points_to_topdown(scene.sim.pathfinder, grid_points[2], meters_per_pixel)
    noise_music_xy_point = [np.round(point).astype(int) for point in noise_music_xy_point]

    spks_start_point = convert_points_to_topdown(scene.sim.pathfinder, [grid_points[0][i][0] for i in range(len(grid_points[0]))], meters_per_pixel)
    spks_start_point = [np.round(point).astype(int) for point in spks_start_point]
    
    display_map(top_down_map, key_points=mic_xy_point+spks_start_point+noise_music_xy_point, filename=filename)


def random_select_start_end_points(scene: Scene, distance_threshold: float = 5.0) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Randomly select start and end points
    """
    # Randomly select start and end index
    start_point = scene.sim.pathfinder.get_random_navigable_point()
    end_point = scene.sim.pathfinder.get_random_navigable_point()
    tries = 0
    while np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[2] - end_point[2]) ** 2) < distance_threshold and\
                    abs(start_point[1] - end_point[1]) > 2:
        if tries > 100:
            end_point = scene.sim.pathfinder.get_random_navigable_point_near(start_point, radius=20.0)
            if np.sqrt((start_point[0] - end_point[0]) ** 2 + (start_point[2] - end_point[2]) ** 2) < distance_threshold:
                continue
            break
        end_point = scene.sim.pathfinder.get_random_navigable_point()
        tries += 1
    return start_point, end_point

def get_nav_idx(scene: Scene,
                distance_threshold: float = 5.0
                ) -> T.Tuple[int, int]:
    """
    Randomly select start and end points and get the navigation index
    """
    
    found_path = False
    while not found_path:
        start_points, end_points = random_select_start_end_points(scene, distance_threshold)
        path = habitat_sim.ShortestPath()
        path.requested_start = start_points
        path.requested_end = end_points
        found_path = scene.sim.pathfinder.find_path(path)
        nav_points = path.points
    
    return nav_points
    
def get_nav_point_from_grid_points(scene: Scene,
                                   grid_points: T.List[T.Tuple[float, float, float]],
                                   distance_threshold: float = 5.0,
                                   num_points: int = 1
                                   ) -> T.List[T.Tuple[float, float, float]]:
    """
    Generate the num_points of navigation points from grid_points
    """
    
    unique_nav_points = []
    find_idx = 0

    while len(unique_nav_points) < num_points:
        nav_point = scene.sim.pathfinder.get_random_navigable_point(max_tries=30)
        close_points_count = 0
        
        for grid_point in grid_points:
            print(f'nav_point: {nav_point}, grid_point: {grid_point}')
            if np.sqrt((nav_point[0] - grid_point[0]) ** 2 + (nav_point[2] - grid_point[2]) ** 2) < distance_threshold and \
               abs(nav_point[1] - grid_point[1]) < 2:
                close_points_count += 1

        if close_points_count >= 2:
            unique_nav_points.append(nav_point)
        
        if find_idx > 500:
            for _ in range(num_points - len(unique_nav_points)):
                random_grid_point = grid_points[np.random.randint(len(grid_points))]
                offset = np.random.uniform(-distance_threshold, distance_threshold, size=2)
                random_point = np.array([random_grid_point[0] + offset[0], random_grid_point[1], random_grid_point[2] + offset[1]])
                random_point = scene.sim.pathfinder.snap_point(random_point)
                if not np.any(np.isnan(random_point)):
                    unique_nav_points.append(np.array([random_point.x, random_point.y, random_point.z]))
                else:
                    unique_nav_points.append(random_grid_point)
                print(f'random_point: {random_point}')
            break
        
        find_idx += 1

    return unique_nav_points
