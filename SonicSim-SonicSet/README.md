# 🛠️ SonicSim Environment Installation

This README provides step-by-step instructions to set up the data preprocessing environment (SonicSim) using Anaconda and to install SoundSpaces 2.0.

## Step 1️⃣ : Create the Environment  

First, First, install the base `ubuntu package` and the `anaconda environment` using the commands below.

```shell
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
conda create -n SonicSim python=3.9 cmake=3.14.0 -y
conda activate SonicSim
```

## Step 2️⃣ : Install SoundSpaces 2.0

### 2.1 Install Dependencies (habitat-lab/habitat-sim)

Clone and install `habitat-sim`:

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless --audio
cd ..
```

Clone and install `habitat-lab`:

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
cd ..
```

**Note:** Edit the `habitat/tasks/rearrange/rearrange_sim.py` file and remove the 36th line where `FetchRobot` is imported.

```shell
cd habitat-lab/habitat/tasks/rearrange/rearrange_sim.py
```

```python
# Remove the following line
from habitat_sim.robots import FetchRobot, FetchRobotNoWheels
```


### 2.2 Install SoundSpaces

Clone and install `sound-spaces`:

```bash
pip install torch torchvision torchaudio
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

# Generate SonicSet

## Install other dependencies

```shell
pip install pyloudnorm pandas
```


## Prepare the speech, music and sound effects

### Step 1️⃣ : Download Librispeech datasets

Download the LibriSpeech dataset from [download link](https://www.openslr.org/12) and put it in the `LibriSpeech/` directory.

### Step 2️⃣ : Download the music and sound effects from the following links

Download the DnR dataset from [download link](https://zenodo.org/records/6949108#.Y9B37S-B3yI) and put it in the `DnR/` directory.

### Step 3️⃣ : Download the Matterport3D

Download the Matterport3D dataset from [download link](https://niessner.github.io/Matterport/) and put it in the `Matterport3D/` directory.

Please note that the Matterport3D dataset is not publicly available. You need to request access to the dataset from the authors.

Refer link: https://github.com/niessner/Matterport

## Generate the SonicSet dataset

To generate the SonicSet dataset, follow the instructions:
```shell
python SonicSet.py # Generate the training, validation and test datasets
```