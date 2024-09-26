# 🛠️ Data Preprocessing Environment Installation

This README provides step-by-step instructions to set up the data preprocessing environment using Anaconda and to install SoundSpaces 2.0.

## Step 1️⃣ : Create the Conda Environment

First, use Anaconda to create the environment based on the provided YAML file.

```bash
conda env create -f SonicSim-SonicSet/data-script/ss-2.0.yaml
conda activate ss-2.0
```

## Step 2️⃣ : Install SoundSpaces 2.0

### 2.1 Install Dependencies (habitat-lab/habitat-sim)

Clone and install `habitat-sim`:

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio
```

Clone and install `habitat-lab`:

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
```

**Note:** Edit the `habitat/tasks/rearrange/rearrange_sim.py` file and remove the 36th line where `FetchRobot` is imported.

### 2.2 Install SoundSpaces

Clone and install `sound-spaces`:

```bash
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

### 2.3 Download Scene Datasets (Matterport3D (MP3D) dataset)
Details: [https://niessner.github.io/Matterport/](https://niessner.github.io/Matterport/). Github: [https://github.com/niessner/Matterport](https://github.com/niessner/Matterport)

We provide 1 example scene from MP3D for performing unit tests in habitat-sim. This can be programmatically downloaded via Habitat's data download utility.
```shell
python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
```

The full MP3D dataset for use with Habitat can be downloaded using the official Matterport3D download script as follows: `python download_mp.py --task habitat -o path/to/download/`. Note that this download script requires python 2.7 to run.

You only need the habitat zip archive and not the entire Matterport3D dataset.

Once you have the habitat zip archive, you should download [this SceneDatasetConfig file](http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json) and place it in the root directory for the Matterport3D dataset (e.g. Habitat-Sim/data/scene_datasets/mp3d/).

### 2.4 SoundSpaces 2.0 Specific Instructions

SoundSpaces 2.0 provides the ability to render IRs on the fly.

#### Download Material Configuration File

Download the material configuration file:

```bash
cd data
wget https://raw.githubusercontent.com/facebookresearch/rlr-audio-propagation/main/RLRAudioPropagationPkg/data/mp3d_material_config.json
```

#### Building GLIBC

For systems with GLIBC version below 2.29, follow the [instructions](https://github.com/facebookresearch/rlr-audio-propagation/issues/9#issuecomment-1317697962) to build the required GLIBC binaries.

To test the installation, run:

```bash
python examples/minimal_example.py
```

Expected output: `data/output.wav` without an error.

### Current Open Issues

- If you encounter [invalid pointer issues](https://github.com/facebookresearch/habitat-sim/issues/1747), import `quaternion` before `habitat_sim` as a workaround.
- If the audio rendering crashes due to errors in loading semantic annotations, set `audio_sensor_spec.enableMaterials = False`.

By following these steps, you should have a fully functional environment for data preprocessing and using SoundSpaces 2.0.

## Generate SonicSim

To generate the SonicSim dataset, follow the instructions:
```shell
python SonicSet_train.py # Generate the training dataset
python SonicSet_val_test.py # Generate the validation and test datasets
```