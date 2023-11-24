### Generic-purposed CAV Trajectory Planning

+ Sampling space should **match** search space

  + translate the X-Y coordinate to frenet coordinate (S-D), S signifies the longitudinal movement and D is the lateral offset;

  + Apply quadratic polynomial function (rather than quintical polynomial) to sample possible trajectories without $s_e$ as a marginal condition and set $v_{d_e}$ to 0;