from collections import deque, namedtuple
import itertools
import os
import random

from moviepy.editor import ImageSequenceClip
import numpy as np
import torch
from typing import List

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'real_done', 'prev_reward', 'arm'))
Transition.__new__.__defaults__ = (None,) * len(Transition._fields) # pre 3.7  
# https://stackoverflow.com/questions/11351032/named-tuple-and-default-values-for-optional-keyword-arguments


class MeanStdevFilter():
    def __init__(self, shape: List, clip: float = 3.0) -> None:
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = np.zeros(shape)
        self.stdev = np.ones(shape) * self.eps

    def update(self, x: np.ndarray) -> None:
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x: np.ndarray) -> np.ndarray:
        return (x * self.stdev) + self.mean


class ReplayPool:

    def __init__(self, capacity: int = int(1e6)) -> None:
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))
        
    def push(self, transition: Transition) -> None:
        """ Saves a transition """
        self._memory.append(transition)
        
    def sample(self, batch_size: int, unique: bool = True) -> Transition:
        transitions = random.sample(self._memory, batch_size) if unique else random.choices(self._memory, k=batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return transitions

    def get_all(self) -> Transition:
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()

    def initialise(self, old_pool: 'ReplayPool'):
        old_memory = old_pool.get_all()
        self._memory.extend(old_memory)


# Code courtesy of JPH: https://github.com/jparkerholder
def make_gif(policy, env, step_count, state_filter, maxsteps=1000, name=None):
    envname = env.spec.id
    if name is None: gif_name = '_'.join([envname, str(step_count)]);
    else: gif_name = str(step_count) + name;
    state = env.reset()
    done = False
    steps = []
    rewards = []
    t = 0
    while (not done) & (t < maxsteps):
        s = env.render('rgb_array')
        steps.append(s)
        action = policy.get_action(state, state_filter=state_filter, deterministic=True)
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        action = action.reshape(len(action), )
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    clip.write_gif('gifs/{}.gif'.format(gif_name), fps=30)


def make_checkpoint(agent, step_count: int, env_name: str) -> None:
    q_funcs, target_q_funcs, policy, log_alpha = agent.q_funcs, agent.target_q_funcs, agent.policy, agent.log_alpha
    
    save_path = "checkpoints/model-{}-{}.pt".format(step_count, env_name)

    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')

    torch.save({
        'double_q_state_dict': q_funcs.state_dict(),
        'target_double_q_state_dict': target_q_funcs.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'log_alpha_state_dict': log_alpha
    }, save_path)


def calculate_huber_loss(td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(
    td_errors: torch.Tensor,
    taus: torch.Tensor,
    weights: torch.Tensor = None,
    kappa: float = 1.0) -> torch.Tensor:

    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


def compute_wd_quantile(q1: torch.Tensor, q2: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """compute 1D gammma-Wasserstein distance 

    Args:
        q1 (tensor): quantiles of first distribution
        q2 (tensor): quantiles of second distribution
        gamma (float, optional): WD-gamma scale. Defaults to 1.0.
    """
    wd = torch.pow(torch.sum(torch.pow(torch.abs(q1 - q2), gamma)), 1 / gamma)
    wd = wd.mean()
    return wd

