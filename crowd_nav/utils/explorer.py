import logging
import copy
import torch
from crowd_nav.utils.state import JointState
from social_gym.src.info import *

class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        if not imitation_learning: self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        
        # Save states and rewards in an output file for benchmarking
        all_states = []
        all_rewards = []

        for i in range(k):
            ob, _ = self.env.reset(phase=phase)
            terminated = False
            truncated = False
            states = []
            actions = []
            rewards = []

            # Save states and rewards in an output file for benchmarking
            states_to_be_saved = []

            while (not terminated) and (not truncated):
                if imitation_learning:
                    state = JointState(self.robot.get_full_state(), ob)
                    states.append(state)

                    # Save states and rewards in an output file for benchmarking
                    if i < 100:
                        states_to_be_saved.append([state.self_state.px - 7.5, state.self_state.py - 7.5, state.self_state.vx, state.self_state.vy])
                        for human_state in state.human_states: states_to_be_saved.append([human_state.px - 7.5, human_state.py - 7.5, human_state.vx, human_state.vy])

                    ob, reward, terminated, truncated, info = self.env.imitation_learning_step()
                else:
                    action = self.robot.act(ob)
                    ob, reward, terminated, truncated, info = self.env.step(action)
                    states.append(self.robot.policy.last_state)
                    actions.append(action)
                rewards.append(reward)
                self.env.render()
                if isinstance(info[0], Danger):
                    too_close += 1
                    min_dist.append(info[0].min_dist)

            # Save states and rewards in an output file for benchmarking
            if i < 100 and imitation_learning:
                all_states.append(states_to_be_saved)
                all_rewards.append(rewards)
                if i == 99: 
                    print('SAVING VARIABLES')
                    import csv, os
                    with open(os.path.join(os.path.dirname(__file__),'data.csv'), 'w') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerows(all_states)
                        writer.writerows(all_rewards)
                        f.close()
                    del all_states
                    del all_rewards

            if isinstance(info[0], ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info[0], Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info[0], Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info[0], ReachGoal) or isinstance(info[0], Collision): # or isinstance(info[0], Timeout)
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.desired_speed)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.desired_speed) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.desired_speed)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list: return sum(input_list) / len(input_list)
    else: return 0
