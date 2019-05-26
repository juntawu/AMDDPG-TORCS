# 预训练模型:
trained from: 
actormodel.h5
criticmodel.h5

# 训练赛道
Train on "Aalborg", 4000 episodes 

# 训练平台
GPU Server，text mode

# 程序版本及回报函数
ddpg_2018_03_28

```
alpha = 180  # max sp passing curve
belta = 40  # expected sp passing curve
if(np.abs(obs['track'][9]) <= 50):
    trackEdgeRatio_9 = (np.abs(obs['track'][9]) / 50)
else:
    trackEdgeRatio_9 = 1
if(np.abs(obs['track'][9]) <= 10):
    spRatio = (1 - np.abs(sp - belta) / alpha)  ## maxmize (spRatio * sp) leads to sp=(alpha+belta)/2
else:
    spRatio = 1
progress = sp * np.cos(obs['angle']) * ( 1 - np.abs(np.sin(obs['angle']) ) ) * ( 1 -  np.abs(obs['trackPos']) ) * trackEdgeRatio_9 * spRatio
```
Note "sp" indicates "speedX"


# 程序设置
* 间隔重启 TORCS 的设置；
* 最大圈数 = 10
* 每训练100回合，测试模型，保存可以通过赛道的模型
* 设置回报阈值

# 问题
* 3450回合的结果能跑完训练赛道Aalborg与测试赛道 CG1, CG2, CG3, E-track4，总体而言，是目前能得到的最好效果
* 4000回合的结果能跑完训练赛道Aalborg与测试赛道 CG1, CG2, CG3, E-track4，但学到的策略是局部最优解，小车在左转弯时表现还行，但是右转弯时会很靠近车道外侧边缘
* 4000回合时在E-Track4中直线高速漂移出车道
* 直线部分仍旧会有少量摇摆状况
* 启动时效果不稳定


# 训练时间
Total training time: 16.302400 hours
