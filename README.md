## RL-TORCS


### Paper
Junta Wu, Huiyun Li. [Aggregated Multi-deep Deterministic Policy Gradient for Self-driving Policy](https://link.springer.com/chapter/10.1007/978-3-030-05081-8_13), International Conference on Internet of Vehicles. Springer, Cham, 2018: 179-192.


### Introduction 
AMDDPG is a deep reinforcement learning algorithm designed based on multi-DDPG structure. This repository tested AMDDPG on [Gym-TORCS](https://github.com/ugo-nama-kun/gym_torcs) and compared it with DDPG.


### Requirements
- Ubuntu 16.04
- Python 2.7
- Numpy, Matplotlib, OpenCV
- Gym
- Keras 1.1.0
- Tensorflow 0.12.0 (CPU version or GPU version)
- [Gym-TORCS](https://github.com/ugo-nama-kun/gym_torcs)
- CUDA (unnecessary if cpu-version tensorflow is installed, no more than CUDA 8.0)



### Environment Configuration
For convenience, environment configuration is done on Anaconda. Terminal commands are shown below.

1. Follow [installation instructions](https://www.anaconda.com/distribution/) to install Anaconda.

2. Create Python 2.7 virtual environment on Anaconda
    '''Shell
    conda create --name python2.7 python=2.7
    source activate python2.7
    '''
    
3. Install Numpy, Matplotlib, OpenCV, h5py in "python2.7" environment
    '''Shell
    pip install --upgrade setuptools
    pip install -U --pre numpy Matplotlib
    pip install -U --pre opencv-python
    pip install h5py
    '''

4. Install Gym
    '''Shell
    pip install -U --pre gym
    '''
    
5. Install Kears
    '''Shell
    pip install -U --pre keras==1.1.0
    '''

6. Install Tensorflow (CPU version)
    '''Shell
    pip install -U --pre tensorflow==0.12.0
    '''
    
7. Install Gym-Torcs
    '''Shell
    git clone https://github.com/ugo-nama-kun/gym_torcs
    sudo apt-get install xautomation libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev 
    cd $Gym-Torcs-ROOT/vtorcs-RL-colors/
    ./configure
    sudo make 
    sudo make install
    sudo make datainstall
    '''
    
8. Clone AMDDPG-TORCS repository
    '''Shell
    git clone 
    
    '''


