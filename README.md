## Tactical Optimistic and Pessimistic estimation (TOP)

Implementation of TOP, an off-policy deep actor-critic framework for continuous control, from our paper [Tactical Optimism and Pessimism for Deep Reinforcement Learning](https://arxiv.org/abs/2102.03765). 

![](extras/ant.gif)



**Running Mujoco:**

```python
python train_top_agent.py
```

We've also included the saved runs across 10 seeds for each environment from the paper in the ```runs``` folder. Each file contains the reward curves used for Figure 3, and is structured as a 10 x 1000 matrix, with each row representing a different seed. 

TOP-TD3 is built on top of the fantastic [TD3 implementation](https://github.com/fiorenza2/TD3_PyTorch) by Philip Ball. 

**Running DM Control Suite**

```python
python top_train.py
```

TOP-RAD is built on top of the original [RAD implementation](https://github.com/MishaLaskin/rad) by Misha Laskin--the majority of the files are unchanged from the original repository. 

We plan to add the saved training data from the DM Control experiments (as we have for the Mujoco experiments) soon! 



Requirements:

- [PyTorch](https://pytorch.org/) >= 1.6.0
- [Tensorboard](https://www.tensorflow.org/tensorboard)
- [Mujoco_py](https://github.com/openai/mujoco-py) >= 2.0.2.13 (Mujoco only)
- [OpenAI Gym](https://gym.openai.com/) >= 0.15.7 
- [DM Control suite](https://github.com/deepmind/dm_control) (DM Control only)
- [dmc2gym](https://github.com/denisyarats/dmc2gym) (DM Control only)
