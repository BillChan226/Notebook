## Probabilistic Safe Policy Search for Inverse Optimal Control (PS2)

### Features

+ **Intrinsic probabilistic constraint** (safety) satisfaction estimation for RL 
+ A better trade-off between optimization objectives using Kalman filter (regions that satisfy the constraint will result in a distribution of high variance, which, after gaussian fusion will have little impact on the original goal distribution)
+ Plug-in-and-play for arbitrary constraint (the original policy will not be flushed off by new constraint)
+ Propose a safety probability estimation metric using value network approximation of advantage function
+ The task policy and safety policy can be learnt from different observation space

### Experiment Design

**PS2 for RL** (Explicit Reward)

**Learning Curve:**

1. doggo - standard
2. Jaka-goal1 - dynamic obstacles -raw pixel
3. inmoov - dynamic obstacles -joint

**Control Accuracy**:

1. **PS2**
2. CPO
3. Lagrangian-PPO
4. Lagrangian-TRPO
5. RMPC
6. RRT

**PS2 for IRL**
**Learning Safety Definition (Cost)**

1. doggo

1. Jaka-goal1 - dynamic obstacles -raw pixel
2. Jaka-goal1 - real

**Learning Curve:**

1. Jaka-goal - dynamic obstacles -sim -IOC -joints -defined cost
2. Jaka-goal - dynamic obstacles -sim -IOC -joints -learned cost

**Control Accuracy**:

1. PS2-IOC
2. CPO-IOC
3. Lagrangian-PPO-IOC
4. Lagrangian-TRPO-IOC
5. RMPC
6. RRT



+ **Simulation**
  + Kuka-button with static obstacle
  + Kuka-button with dynamic obstacle
  + Jaka-tomato with static obstacle
  + Jaka-tomato with dynamic obstacle
  + InMoov-tomato with static obstacle
  + InMoov-tomato with moving obstacle
+ **Real-life**
  + Jaka-reach with static obstacle
  + Jaka-reach with dynamic obstacle (human in the same workspace)
  + InMoov-tomato with static obstacle
  + InMoov-tomato with dynamic obstacle (human in the same workspace)



**PS2 for IRL** (Implicit Reward)

+ **Simulation**
  + Kuka-button with static obstacle
  + Kuka-button with dynamic obstacle
  + Jaka-tomato with static obstacle
  + Jaka-tomato with dynamic obstacle
  + InMoov-tomato with static obstacle
  + InMoov-tomato with moving obstacle
+ **Real-life**
  + Jaka-reach with static obstacle
  + Jaka-reach with dynamic obstacle (human in the same workspace)
  + InMoov-tomato with static obstacle
  + InMoov-tomato with dynamic obstacle (human in the same workspace)



Model Path:

RL1: /home/gong112/service_backup/work/RL-InmoovRobot/data/jaka_live_ppo

RL2: /home/gong112/service_backup/work/RL-InmoovRobot/data/jaka_live_ppo2

IRL: /home/gong112/service_backup/work/RL-InmoovRobot/data/gcl_live_jaka

IRL2: /home/gong112/service_backup/work/RL-InmoovRobot/data/gcl_live_jaka_new



**Experiment**:

**Algo Comparison**:

+ PPO

+ CPO

+ **CSC**
+ **SAILR**

+ **PPO-ISSA**

+ PPO-Lagrangian

+ **PPO-CBF**

+ APS (Ours)



**Metrics**:

+ **Normalized return**

+ **Normalized constraint violation**

+ **normalized cost rate**
+ Inference time (e.g. sampling time)
+ Safety Violation



**Environment**:

+ Standard Safety-gym: SG6 (SG18)
+ Jaka-Joint
+ Jaka-Raw
+ Inmoov-Joint
+ Kuka-Joint-Dynamics



Env id:

Jaka: JakaSafeGymEnv-v1

Kuka: KukaButtonGymEnv

InMoov: InmoovSafeGymEnv-v0

SG6: 

PointGoal1

PointGoal2

PointButton1

PointPush1

CarGoal1

DoggoGoal1



Algos:

+ PPO
+ PPO-Lagrangian
+ CPO
+ **CSC**
+ **SAILR**
+ **PPO-ISSA**
+ USL
+ FAC
+ Recovery RL
+ Safety Layer
+ APS (Ours)



Adversarial Policy: unconstrained_TD3_JakaSafeGymEnv-v5_0



