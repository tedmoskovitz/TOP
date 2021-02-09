## Dynamic Optimistic and Pessimistic Estimation (DOPE)

Implementation of DOPE, an off-policy deep actor-critic algorithm for continuous control, from our paper [Deep Reinforcement Learning with Dynamic Optimism](https://arxiv.org/abs/2102.03765). 

![](extras/ant.gif)

To run:

```python
python train_dope_agent.py
```



We've also included the saved runs across 10 seeds for each environment from the paper in the ```runs``` folder. Each file contains the reward curves used for Figure 3, and is structured as a 10 x 1000 matrix, with each row representing a different seed. 



Built on top of the fantastic [TD3 implementation](https://github.com/fiorenza2/TD3_PyTorch) by Philip Ball. 



Requirements:

- [PyTorch](https://pytorch.org/) >= 1.6.0
- [Tensorboard](https://www.tensorflow.org/tensorboard)
- [Mujoco_py](https://github.com/openai/mujoco-py) >= 2.0.2.13
- [OpenAI Gym](https://gym.openai.com/) >= 0.15.7



If you find this code useful, it would be great if you could cite us using: 

```
@misc{moskovitz2021deep,
      title={Deep Reinforcement Learning with Dynamic Optimism}, 
      author={Ted Moskovitz and Jack Parker-Holder and Aldo Pacchiano and Michael Arbel},
      year={2021},
      eprint={2102.03765},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

