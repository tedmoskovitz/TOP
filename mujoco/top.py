import copy

import numpy as np
import torch
import pdb
import torch.nn.functional as F

from utils import ReplayPool, calculate_quantile_huber_loss, compute_wd_quantile
from networks import Policy, QuantileDoubleQFunc
from bandit import ExpWeights
from typing import Callable, Dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TOP_Agent:

    def __init__(
        self,
        seed: int,
        state_dim: int,
        action_dim: int,
        action_lim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 5e-3,
        batchsize: int = 256,
        hidden_size: int = 256,
        update_interval: int = 2,
        buffer_size: int = int(1e6),
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        explore_noise: float = 0.1,
        n_quantiles: int = 100,
        kappa: float = 1.0,
        beta: float = 0.0,
        bandit_lr: float = 0.1
        ) -> None:
        """
        Initialize DOPE agent. 

        Args:
            seed (int): random seed
            state_dim (int): state dimension
            action_dim (int): action dimension
            action_lim (int, optional): max action value. Defaults to 1.
            lr (float, optional): learning rate. Defaults to 3e-4.
            gamma (float, optional): discount factor. Defaults to 0.99.
            tau (float, optional): mixing rate for target nets. Defaults to 5e-3.
            batchsize (int, optional): batch size. Defaults to 256.
            hidden_size (int, optional): hidden layer size for policy. Defaults to 256.
            update_interval (int, optional): delay for actor, target updates. Defaults to 2.
            buffer_size (int, optional): size of replay buffer. Defaults to int(1e6).
            target_noise (float, optional): smoothing noise for target action. Defaults to 0.2.
            target_noise_clip (float, optional): limit for target. Defaults to 0.5.
            explore_noise (float, optional): noise for exploration. Defaults to 0.1.
            n_quantiles (int, optional): number of quantiles. Defaults to 100.
            kappa (float, optional): constant for Huber loss. Defaults to 1.0.
            bandit_lr (float, optional): bandit learning rate. Defaults to 0.1.
        """
        self.gamma = gamma
        self.tau = tau
        self.batchsize = batchsize
        self.update_interval = update_interval
        self.action_lim = action_lim

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

        torch.manual_seed(seed)

        # init critic(s)
        self.q_funcs = QuantileDoubleQFunc(state_dim, action_dim, n_quantiles=n_quantiles, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # init actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False

        # set distributional parameters
        taus = torch.arange(
            0, n_quantiles+1, device=device, dtype=torch.float32) / n_quantiles
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, n_quantiles)
        self.n_quantiles = n_quantiles
        self.kappa = kappa

        # bandit top-down controller
        self.TDC = ExpWeights(arms=[-1, 0], lr=bandit_lr, init=0.0, use_std=True) 

        # init optimizers
        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = ReplayPool(capacity=int(buffer_size))

        self._update_counter = 0

    def reallocate_replay_pool(self, new_size: int) -> None:
        """Reset buffer

        Args:
            new_size (int): new maximum buffer size. 
        """
        assert new_size != self.replay_pool.capacity, "Error, you've tried to allocate a new pool which has the same length"
        new_replay_pool = ReplayPool(capacity=new_size)
        new_replay_pool.initialise(self.replay_pool)
        self.replay_pool = new_replay_pool

    def get_action(
        self, 
        state: np.ndarray,
        state_filter: Callable = None,
        deterministic: bool = False) -> np.ndarray:
        """given the current state, produce an action

        Args:
            state (np.ndarray): state input. 
            state_filter (Callable): pre-processing function for state input. Defaults to None.
            deterministic (bool, optional): whether the action is deterministic or stochastic. Defaults to False.

        Returns:
            np.ndarray: the action. 
        """
        if state_filter:
            state = state_filter(state)
        state = torch.Tensor(state).view(1,-1).to(device)
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze().cpu().numpy())
    
    def update_target(self) -> None:
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_functions(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        nextstate_batch: torch.Tensor,
        done_batch: torch.Tensor,
        beta: float) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """compute quantile losses for critics

        Args:
            state_batch (torch.Tensor): batch of states
            action_batch (torch.Tensor): batch of actions
            reward_batch (torch.Tensor): batch of rewards
            nextstate_batch (torch.Tensor): batch of next states
            done_batch (torch.Tensor): batch of booleans describing whether episode ended. 
            beta (float): optimism parameter

        Returns:
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                critic 1 loss, critic 2 loss, critic 1 quantiles, critic 2 quantiles
        """
        with torch.no_grad():
            # get next action from target network
            nextaction_batch = self.target_policy(nextstate_batch)
            # add noise
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            # get quantiles at (s', \tilde a)
            quantiles_t1, quantiles_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # compute mean and std
            quantiles_all = torch.stack([quantiles_t1, quantiles_t2], dim=-1) # [batch_size, n_quantiles, 2]
            mu = torch.mean(quantiles_all, axis=-1) # [batch_size, n_quantiles]
            # compute std by hand for stability 
            sigma = torch.sqrt((torch.pow(quantiles_t1 - mu, 2) + torch.pow(quantiles_t2 - mu, 2)) + 1e-4) 
            # construct belief distribution
            belief_dist = mu + beta * sigma  # [batch_size, n_quantiles]
            # compute the targets as batch_size x 1 x n_quantiles
            n_quantiles = belief_dist.shape[-1]
            quantile_target = reward_batch[..., None] + (1.0 - done_batch[..., None]) \
                * self.gamma * belief_dist[:, None, :] # [batch_size, 1, n_quantiles]
        
        # get quantiles at (s, a)
        quantiles_1, quantiles_2 = self.q_funcs(state_batch, action_batch)
        # compute pairwise td errors
        td_errors_1 = quantile_target - quantiles_1[..., None] # [batch_size, n_quantiles, n_quantiles]
        td_errors_2 = quantile_target - quantiles_2[..., None] # [batch_size, n_quantiles, n_quantiles]
        # compute quantile losses
        loss_1 = calculate_quantile_huber_loss(td_errors_1, self.tau_hats, weights=None, kappa=self.kappa)
        loss_2 = calculate_quantile_huber_loss(td_errors_2, self.tau_hats, weights=None, kappa=self.kappa)

        return loss_1, loss_2, quantiles_1, quantiles_2

    def update_policy(self, state_batch: torch.Tensor, beta: float) -> torch.Tensor:
        """update the actor. 

        Args:
            state_batch (torch.Tensor): batch of states. 
            beta (float): optimism parameter.

        Returns:
            torch.Tensor: DPG loss. 
        """
        # get actions a
        action_batch = self.policy(state_batch)
        # compute quantiles (s,a)
        quantiles_b1, quantiles_b2 = self.q_funcs(state_batch, action_batch)
        # construct belief distribution
        quantiles_all = torch.stack([quantiles_b1, quantiles_b2], dim=-1) # [batch_size, n_quantiles, 2]
        mu = torch.mean(quantiles_all, axis=-1) # [batch_size, n_quantiles]
        eps1, eps2 = 1e-4, 1.1e-4 # small constants for stability 
        sigma = torch.sqrt((torch.pow(quantiles_b1 + eps1 - mu, 2) + torch.pow(quantiles_b2 + eps2 - mu, 2)) + eps1) 
        belief_dist = mu + beta * sigma # [batch_size, n_quantiles]
        # DPG loss
        qval_batch = torch.mean(belief_dist, axis=-1)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    def optimize(
        self,
        n_updates: int,
        beta: float,
        state_filter: Callable = None
        ) -> [float, float, float, float, torch.Tensor, torch.Tensor]:
        """sample transitions from the buffer and update parameters

        Args:
            n_updates (int): number of updates to perform.
            beta (float): optimism parameter.
            state_filter (Callable, optional): state pre-processing function. Defaults to None.

        Returns:
            [float, float, float, float, torch.Tensor, torch.Tensor]:
                critic 1 loss, critic 2 loss, actor loss, WD, critic 1 quantiles, critic 2 quantiles
        """
        q1_loss, q2_loss, wd, pi_loss = 0, 0, 0, None
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)
            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)
            
            # update q-funcs
            q1_loss_step, q2_loss_step, quantiles1_step, quantiles2_step = self.update_q_functions(state_batch, action_batch, reward_batch, nextstate_batch, done_batch, beta)
            q_loss_step = q1_loss_step + q2_loss_step

            # measure wasserstein distance
            wd_step = compute_wd_quantile(quantiles1_step, quantiles2_step)
            wd += wd_step.detach().item()

            # take gradient step for critics
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()
            
            self._update_counter += 1

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()

            # every update_interval steps update actor, target nets
            if self._update_counter % self.update_interval == 0:
                if not pi_loss:
                    pi_loss = 0
                # update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step = self.update_policy(state_batch, beta)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
                # update target policy and q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        return q1_loss, q2_loss, pi_loss, wd / n_updates, quantiles1_step, quantiles2_step
