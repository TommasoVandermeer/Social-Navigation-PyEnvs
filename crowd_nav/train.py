import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gymnasium as gym
from utils.trainer import Trainer
from utils.memory import ReplayMemory
from utils.explorer import Explorer
from policy.policy_factory import policy_factory
from social_gym.src.robot_agent import RobotAgent

import warnings
warnings.filterwarnings('ignore')

def main(env_config_dir, policy_name, policy_config_dir, train_config_dir, output_dir, weights, resume, gpu, debug):
    ## Configure paths
    make_new_dir = True
    if os.path.exists(output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y' and not resume:
            shutil.rmtree(output_dir)
        else:
            make_new_dir = False
            env_config_dir = os.path.join(output_dir, os.path.basename(env_config_dir))
            policy_config_dir = os.path.join(output_dir, os.path.basename(policy_config_dir))
            train_config_dir = os.path.join(output_dir, os.path.basename(train_config_dir))
    if make_new_dir:
        os.makedirs(output_dir)
        shutil.copy(env_config_dir, output_dir)
        shutil.copy(policy_config_dir, output_dir)
        shutil.copy(train_config_dir, output_dir)
    log_file = os.path.join(output_dir, 'output.log')
    il_weight_file = os.path.join(output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(output_dir, 'rl_model.pth')

    ## Configure logging
    mode = 'a' if resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S", force=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    logging.info('Using device: %s', device)

    ## Configure policy
    policy = policy_factory[policy_name]()
    if not policy.trainable:
        parser.error('Policy has to be trainable')
    if policy_config_dir is None:
        parser.error('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_dir)
    policy.configure(policy_config)
    policy.set_device(device)

    ## Configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_dir)
    env = gym.make('SocialGym-v0')
    env.configure(env_config)
    robot = RobotAgent(env)
    robot.configure(env_config, 'robot')
    env.set_robot(robot)

    ## Read training parameters
    if train_config_dir is None:
        parser.error('Train config has to be specified for a trainable network')
    train_config = configparser.RawConfigParser()
    train_config.read(train_config_dir)
    rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
    train_batches = train_config.getint('train', 'train_batches')
    train_episodes = train_config.getint('train', 'train_episodes')
    sample_episodes = train_config.getint('train', 'sample_episodes')
    target_update_interval = train_config.getint('train', 'target_update_interval')
    evaluation_interval = train_config.getint('train', 'evaluation_interval')
    capacity = train_config.getint('train', 'capacity')
    epsilon_start = train_config.getfloat('train', 'epsilon_start')
    epsilon_end = train_config.getfloat('train', 'epsilon_end')
    epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
    checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

    ## Configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)

    ## Set robot policy (RL policy)
    policy.set_env(env)
    robot.set_policy(policy)
    robot.print_info()

    ## Imitation learning
    if resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        il_episodes = train_config.getint('imitation_learning', 'il_episodes')
        il_policy = train_config.get('imitation_learning', 'il_policy')
        il_epochs = train_config.getint('imitation_learning', 'il_epochs')
        il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
        trainer.set_learning_rate(il_learning_rate)
        if robot.visible: safety_space = 0
        else: safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        env.set_safety_space(safety_space)
        env.set_human_motion_model_as_robot_policy(il_policy, runge_kutta=False)
        explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_epochs)
        torch.save(model.state_dict(), il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    explorer.update_target_model(model)

    ## Reinforcement learning
    env.set_safety_space(0)
    trainer.set_learning_rate(rl_learning_rate)
    # fill the memory pool with some RL experience
    if resume:
        robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(100, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
    episode = 0
    while episode < train_episodes:
        if resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        if len(trainer.memory.memory) > 0:
            trainer.optimize_batch(train_batches)
        episode += sample_episodes

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)

    ## Final test
    explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


if __name__ == '__main__':
    # Parse options from command line
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config_dir', type=str, default=os.path.join(os.path.dirname(__file__),'configs/env.config'))
    parser.add_argument('--policy_name', type=str, default='cadrl')
    parser.add_argument('--policy_config_dir', type=str, default=os.path.join(os.path.dirname(__file__),'configs/policy.config'))
    parser.add_argument('--train_config_dir', type=str, default=os.path.join(os.path.dirname(__file__),'configs/train.config'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__),'data/output'))
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    main(**vars(args))
