#!/bin/bash

#SBATCH --job-name=TD3_FORK   # Job name
#SBATCH --partition=ug-gpu-small          # Partition name
#SBATCH --gres=gpu
#SBATCH --mem=8G                 # Memory total in MB (for all cores)
#SBATCH --qos=short
#SBATCH --time=24:00:00          # Time limit hrs:min:sec
#SBATCH --output=TD3_FORK/myPythonJob.log # Standard output and error log

# Create and activate a Python virtual environment
python3 -m venv RL_Kernel

source /home2/cgmj52/ReinforcementLearning/RL_Kernel/bin/activate
# Install necessary packages using pip
pip install numpy matplotlib gym pyvirtualdisplay
pip install torch torchvision
# For PyTorch, use the correct pip command from the PyTorch website, assuming CPU-only here
pip install torch torchvision torchaudio
pip install setuptools==65.5.0 "wheel<0.40.0"
apt update
apt-get install python3-opengl
apt install xvfb -y
pip install 'swig'
pip install 'pyglet==1.5.27'
pip install 'pyvirtualdisplay==3.0'
pip install 'gym[box2d]==0.20.0'
# Now run Python script
python3 /home2/cgmj52/ReinforcementLearning/TD3_FORK/train_RL.py

# Deactivate the virtual environment at the end
deactivate
