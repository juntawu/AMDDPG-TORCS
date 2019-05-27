# 预训练模型:
trained from: 
original actormodel and criticmodel

# 训练赛道
Train on "Aalborg", 6000 episodes 

# 训练平台
GPU Server(2 GPU), text mode，

# 程序版本及回报函数
multi_ddpg_2018_05_10

```
alpha = 180  # max sp passing curve
belta = 40  # expected sp passing curve
if(np.abs(obs['track'][9]) <= 50):
    trackEdgeRatio_9 = (np.abs(obs['track'][9]) / 50)
else:
    trackEdgeRatio_9 = 1
if(np.abs(obs['track'][9]) <= 15):
    spRatio = (1 - np.abs(sp - belta) / alpha)  ## maxmize (spRatio * sp) leads to sp=(alpha+belta)/2
else:
    spRatio = 1
progress = sp * np.cos(obs['angle']) * ( 1 - np.abs(np.sin(obs['angle']) ) ) * ( 1 -  np.abs(obs['trackPos']) ) * trackEdgeRatio_9 * spRatio
```
Note "sp" indicates "speedX"


# 训练方案
每个episode选择一个网络，由网络给定动作，与TORCS交互，并训练输出动作的网络，所得到的经验放入公共的经验回放池中，即各个子网络的经验回放池是一样的。模型测试时，将多个网络的输出进行平均作为最终策略。


# 程序设置
* 间隔重启 TORCS 的设置；
* 最大圈数 = 50
* 每训练50回合，测试模型，保存可以通过赛道的模型
* 设置回报阈值
* 每回合最大训练步数为5000

# 效果
* 

# 问题


# 训练时间
27.846
