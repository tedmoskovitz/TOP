import random
from argparse import ArgumentParser
from collections import deque

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch
import pdb
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Callable

from top import TOP_Agent
from utils import MeanStdevFilter, Transition, make_gif, make_checkpoint

GYM_ENV = gym.wrappers.time_limit.TimeLimit

def train_agent_model_free(agent: DOPE_Agent, env: GYM_ENV, params: Dict) -> None:
    
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 1000
    gif_interval = 1000000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']
    save_model = params['save_model']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps

    com = f"DOPE_{params['env']}_nq{params['n_quantiles']}_{params['bandit_lr']}_seed{seed}"
    writer = SummaryWriter(log_dir='dope_runs/' + com)

    prev_episode_reward = 0
    while samples_number < 1e6:
        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False

        # sample an optimism setting for this episode
        optimism = agent.TDC.sample()

        while (not done):
            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            
            nextstate, reward, done, _ = env.step(action)
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
            state = nextstate
            if state_filter:
                state_filter.update(state)
            episode_reward += reward
            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                q1_loss, q2_loss, pi_loss, avg_wd, q1, q2 = agent.optimize(update_timestep, optimism, state_filter=state_filter)
                n_updates += 1
            # logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
                writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
                writer.add_scalar('Loss/WD', avg_wd, n_updates)
                writer.add_scalar('Distributions/Mean_1', torch.mean(q1), n_updates)
                writer.add_scalar('Distributions/Median_1', torch.median(q1), n_updates)
                writer.add_scalar('Distributions/Mean_2', torch.mean(q2), n_updates)
                writer.add_scalar('Distributions/Median_2', torch.median(q2), n_updates)

                # bandit tracking
                writer.add_scalar('Distributions/optimism', optimism, n_updates)
                arm_probs = agent.TDC.get_probs() 
                for i, p in enumerate(arm_probs):
                    writer.add_scalar(f'Distributions/arm{i}', p, n_updates)

                if pi_loss:
                    writer.add_scalar('Loss/policy', pi_loss, n_updates)
                avg_length = np.mean(episode_steps)
                running_reward = np.mean(episode_rewards)
                eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals)
                writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
                writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
                progress_str = f'Episode {i_episode} \t Samples {samples_number} \t Avg length: {avg_length} \t Test reward: {eval_reward} \t Train reward: {running_reward} \t WD: {avg_wd} \t Number of Updates: {n_updates}'
                print(progress_str)
                episode_steps = []
                episode_rewards = []
            if cumulative_timestep % gif_interval == 0:
                make_gif(agent, env, cumulative_timestep, state_filter, name=com)
                if save_model:
                    make_checkpoint(agent, cumulative_timestep, params['env'])

        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)

        # update bandit parameters
        feedback = episode_reward - prev_episode_reward
        agent.TDC.update_dists(feedback)
        prev_episode_reward = episode_reward


def evaluate_agent(
    env: GYM_ENV,
    agent: DOPE_Agent,
    state_filter: Callable,
    n_starts: int = 1) -> float:
    
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=25000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--n_quantiles', type=int, default=50)
    parser.add_argument('--bandit_lr', type=float, default=0.1)
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env = gym.make(params['env'])
    env = RescaleAction(env, -1, 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # initialize agent
    agent = DOPE_Agent(seed, state_dim, action_dim, \
        n_quantiles=params['n_quantiles'], bandit_lr=params['bandit_lr'])

    # train agent 
    train_agent_model_free(agent=agent, env=env, params=params)


if __name__ == '__main__':
    main()
