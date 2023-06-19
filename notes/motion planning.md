# Motion Planning

Alothough this motion planning chapter mainly aims at robotics application, I try to make the approaches and algorithms as general as possible.

**Rubrics**

+ computationally efficient (real-time demand)
+ scale to high-dimensional setting
+ generalizability to unseen workspace scenarios



**Mainstream Approaches**

+ Traditionnal Motion Planner
  + Sampling-based motion planning (SMP)
    + Rapidly-exploring Random Trees (RRT)
    + Optimal RRT (RRT*)
    + Potentially guided-RRT* (P-RRT*)

+ Deep Reinforcement Learning (DRL)
  + rely on exploraion through interaction with the environment, thus hampering its applications to real-world robotic scenarios
+ Imitation Learning (Learning from Demonstration)
  + Behavior Cloning
  + Inverse Reinforcement Learning



## Imitation Learning

### MPNet: A Neural Motion Planner

MPNet is a neural network based motion planner comprised of two phases: 

+ offline training of the neural models;
+ online path generation.



#### Offline Training

This phrase encloses two neural models. 

##### state encoder

The first model (Enet) is an encoder that embeds the obstacles point cloud ($X_{obs}$) into a latent space.

Training strategy:

+ Encoder-decoder architecture (contrative autoencoder(CAE)) 

  Reconstruction loss:
  $$
  L_{AE}(\theta^e,\theta^d)=\frac{1}{N_{obs}}\sum_{x \in D_{obs}}||x-\hat{x}||^2+\lambda \sum_{ij}(\theta^e_{ij})^2
  $$

+ End-to-end fashion with Pnet

##### planning network

Given the obstacles encoding $Z$, current state $x_t$ and the goal state $x_T$, Pnet predicts the next state $\hat{x}_{t+1}\in X_{free}$.
$$
L_{Pnet}(\theta)=\frac{1}{N_p}\sum^{\hat{N}}_j\sum^{T-1}_{i=0}||x_{j,i+1}-\hat{x}_{j,i+1||^2}
$$
To inculcate stochasticity into the Pnet, some of the hidden units in each of its hidden layer were dropped out with a probability $p:[0,1] \in R$.



#### Online Path Planning

![image-20230307155856398](/Users/zhaorunchen/Library/Application Support/typora-user-images/image-20230307155856398.png)



