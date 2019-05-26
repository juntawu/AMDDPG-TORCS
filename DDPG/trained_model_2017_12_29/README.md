# Pre-trained model
trained from: 
actormodel.h5
criticmodel.h5

# Platform
8-Titan Server(CPU only)

# Track
Train on "CG1" 

# Reward
reward = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
Note "sp" indicates "speedX"

# Issues
* pass the whole "CG1" Track, but do not perform well on other tracks (like "Spring")
