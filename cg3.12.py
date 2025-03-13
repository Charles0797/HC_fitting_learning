import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import openseespy.opensees as ops
import pickle

# ========== 1. ScalingLayer：可学习的输入缩放层 ==========
class ScalingLayer(nn.Module):
    def __init__(self, input_dim):
        super(ScalingLayer, self).__init__()
        # 初始 log_scale = 0 => exp(0) = 1
        self.log_scale = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # 保证缩放因子始终为正
        return x * torch.exp(self.log_scale)

# ========== 2. 环境：HysteresisEnv ==========
class HysteresisEnv:
    """
    环境仍是基于参数 p1, p2 的识别任务。
    target_params = [235, 123, 0.1] 中的前两个维度与当前环境交互，第三个(0.1)仅在 hysteretic_curve()中使用。
    如果想识别更多参数，可自行扩展。
    """
    def __init__(self):
        self.param_ranges = [[200, 400], [100, 200]]  # 两个可调参数的取值范围
        original_protocol = np.array([1., 2., 3., 2., 1., 0., -1., -2., -3., -2., -1., 0.])
        # 使用插值生成 2000 个数据点
        self.protocol = np.interp(
            np.linspace(0, 1, 2000),
            np.linspace(0, 1, len(original_protocol)),
            original_protocol
        )
        # 目标参数 (p1, p2, strainHardeningRatio)
        self.target_params = [235, 123, 0.1]
        # self.boundary_margin = 0.1  # 边界10%区域触发惩罚
        # self.max_penalty = 50  # 最大惩罚值
        self.prev_error = None
        self.reset()

    def random_params(self):
        return [random.uniform(*r) for r in self.param_ranges]

    def hysteretic_curve(self, params):
        """
        用 OpenSees 建一个简单的单元，然后施加 self.protocol 位移，返回力-位移曲线。
        """
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)
        ops.node(1, 0.0)
        ops.node(2, 0.0)
        ops.fix(1, 1)
        # p1= params[0], p2= params[1], strainHardeningRatio= target_params[2]
        ops.uniaxialMaterial("Steel01", 1, params[0], params[1], self.target_params[2])
        ops.element("twoNodeLink", 1, 1, 2, "-mat", 1, "-dir", 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 1.0)
        ops.algorithm("Newton")
        ops.integrator("DisplacementControl", 2, 1, 1)
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("BandSPD")
        ops.analysis("Static")

        result = []
        prev_disp = 0.0
        for disp in self.protocol:
            inc = disp - prev_disp
            prev_disp = disp
            ops.integrator("DisplacementControl", 2, 1, inc)
            if ops.analyze(1) == 0:
                force = ops.eleForce(1)[1]
                result.append(force)
            else:
                # 如果分析失败，则记为0
                result.append(0.0)
        ops.wipe()
        return np.array(result)

    def step(self, action):
        """
        action: [a1, a2], 对 current_params 做加减操作，范围限制在 param_ranges。
        返回 (next_state, reward)
        """
        scale_factors = [1, 1]  # 针对不同参数设置不同更新尺度
        new_params = []
        for i, (p, a, r, s) in enumerate(zip(self.current_params, action, self.param_ranges, scale_factors)):
            new_val = p + s * a
            # clip 到 [r[0], r[1]]
            new_val = max(min(new_val, r[1]), r[0])
            new_params.append(new_val)
        self.current_params = new_params

        # boundary_penalty = 0
        # for i, p in enumerate(new_params):
        #     lower, upper = self.param_ranges[i]
        #     range_width = upper - lower
        #     if p < lower + self.boundary_margin * range_width:
        #         penalty_ratio = (lower + self.boundary_margin * range_width - p) / (self.boundary_margin * range_width)
        #         boundary_penalty += self.max_penalty * penalty_ratio ** 2
        #     elif p > upper - self.boundary_margin * range_width:
        #         penalty_ratio = (p - (upper - self.boundary_margin * range_width)) / (
        #                     self.boundary_margin * range_width)
        #         boundary_penalty += self.max_penalty * penalty_ratio ** 2

        # 奖励设计：对比前两个参数与 target_params 的比值误差
        current_errors = [(p / t - 1) ** 2 for p, t in zip(new_params, self.target_params[:2])]
        current_mse = np.mean(current_errors)

        if self.prev_error is None:
            reward = 100 * np.exp(-current_mse)
            self.prev_error = current_mse
        else:
            delta_error = self.prev_error - current_mse
            direction_reward = np.sign(delta_error) *1000 * np.abs(delta_error)
            absolute_reward = 100 * np.exp(-current_mse)
            reward = 0.7 * absolute_reward + 0.3 * direction_reward
            # reward -= boundary_penalty  # 从总奖励中扣除

            # target_curve = self.hysteretic_curve(self.target_params[:2])
            # current_curve = self.hysteretic_curve(new_params)
            # curve_similarity = np.exp(-0.001 * np.mean((target_curve - current_curve) ** 2))
            #
            # # 组合奖励
            # reward = (
            #         0.4 * absolute_reward +
            #         0.3 * direction_reward +
            #         0.3 * 100 * curve_similarity
            # )
            self.prev_error = current_mse

        # next_state 就是新的 [param1, param2]
        return self.current_params, reward

    def reset(self):
        """
        重置环境状态：随机初始化 current_params
        """
        self.prev_error = None
        self.current_params = self.random_params()
        return self.current_params

# ========== 3. Actor 与 Critic 网络结构：多层全连接 ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.scaling = ScalingLayer(state_dim)
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.scaling(x)
        return self.fc(x)


# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, param_ranges):
#         super().__init__()
#         self.param_ranges = torch.tensor(param_ranges)  # shape: (2,2)
#         self.scaling = ScalingLayer(state_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.LeakyReLU(0.01),
#             nn.LayerNorm(256),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(0.01),
#             nn.LayerNorm(256),
#             nn.Linear(256, action_dim),
#             nn.Tanh()  # 输出范围[-1,1]
#         )
#
#     def forward(self, x):
#         x = self.scaling(x)
#         raw_action = self.fc(x)
#         # 线性映射到参数范围内
#         mid = (self.param_ranges[:, 1] + self.param_ranges[:, 0]) / 2
#         dif = (self.param_ranges[:, 1] - self.param_ranges[:, 0]) / 2
#         action = mid + dif * raw_action
#         return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.scaling = ScalingLayer(state_dim + action_dim)
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        # 将 state, action 拼接后输入
        sa = torch.cat([state, action], dim=1)
        sa = self.scaling(sa)
        q = self.fc(sa)
        return q

# ========== 4. ReplayBuffer ==========
class ReplayBuffer:
    def __init__(self, capacity, filename='replay_buffer.csv'):
        self.buffer = []
        self.capacity = capacity
        self.filename = filename
        self.columns = ['State', 'Action', 'Reward', 'NextState']

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 将每个字段拆开
        states = []
        actions = []
        rewards = []
        next_states = []
        for s, a, r, s_ in batch:
            states.append(s)
            actions.append(a)
            rewards.append([r])  # shape要是(N,1)
            next_states.append(s_)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# ========== 5. DDPG 常用的软更新函数 ==========
def soft_update(target_net, source_net, tau=0.005):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )

# ========== 6. 训练函数：包含 DDPG 更新逻辑 ==========
def train_ddpg(env, actor, critic, actor_optimizer, critic_optimizer,
               buffer, num_episodes=100, max_steps=1000, batch_size=64, gamma=0.99):
    """
    - 引入目标网络 target_actor, target_critic
    - 每一步都把 (s, a, r, s') 存到 buffer，如果 buffer 中数据足够，就采样更新
    """
    # 构建目标网络，并拷贝参数
    target_actor = Actor(2, 2)
    target_critic = Critic(2, 2)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    target_actor.eval()
    target_critic.eval()

    rewards_history = []
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 预先计算目标曲线
    target_curve = env.hysteretic_curve(env.target_params)
    target_min = np.min(target_curve)
    target_max = np.max(target_curve)
    fixed_xlim = (-3.5, 3.5)
    fixed_ylim = (target_min - 50, target_max + 50)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_rewards = []

        # 线性衰减探索
        exploration_noise = max(0.1, 1 - episode / num_episodes)

        for step in range(max_steps):
            # 1. 从 actor 得到动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            # 加上随机噪声
            action += exploration_noise * np.random.normal(size=action.shape)
            # clip 动作到 [-1,1] 范围内
            action = np.clip(action, -1.0, 1.0)

            # 2. 与环境交互
            next_state, reward = env.step(action)
            buffer.add((state, action, reward, next_state))

            episode_reward += reward
            step_rewards.append(reward)
            state = next_state

            # 3. 可视化（每隔若干步）
            if step % 10 == 0:
                fitted_curve = env.hysteretic_curve(env.current_params)
                ax1.clear()
                ax1.plot(env.protocol, target_curve, 'b-', label='Target')
                ax1.plot(env.protocol, fitted_curve, 'r--', label='Current')
                param_info = "/".join([f"{p:.2f}" for p in env.current_params])
                target_info = "/".join([f"{p:.2f}" for p in env.target_params])
                ax1.set_title(f'Ep {episode + 1} Step {step}\nCurrent: {param_info}\nTarget: {target_info}')
                ax1.legend()
                ax1.set_xlim(fixed_xlim)
                ax1.set_ylim(fixed_ylim)

                ax2.clear()
                ax2.plot(step_rewards, 'g-', label='InstantReward')
                window_size = max(1, len(step_rewards) // 10)
                if len(step_rewards) >= window_size:
                    ma = np.convolve(step_rewards, np.ones(window_size) / window_size, mode='valid')
                    ax2.plot(range(window_size - 1, len(step_rewards)), ma, 'b--', label=f'MA({window_size})')
                ax2.legend()
                if step_rewards:
                    min_r = min(step_rewards)
                    max_r = max(step_rewards)
                    range_padding = max(0.5, (max_r - min_r) * 0.3)
                    ax2.set_ylim(min(min_r - range_padding, -1), max(max_r + range_padding, 2))
                else:
                    ax2.set_ylim(-1, 10)
                plt.pause(0.01)

            # 4. 如果 buffer 中数据足够，则进行一次 DDPG 更新
            if len(buffer) > batch_size:
                states_b, actions_b, rewards_b, next_states_b = buffer.sample(batch_size)

                # ---- Critic 更新 ----
                # 4.1 计算 y = r + gamma * Q'(s', a')
                with torch.no_grad():
                    # target_actor(next_state)
                    next_actions = target_actor(next_states_b)
                    next_q_values = target_critic(next_states_b, next_actions)
                    y = rewards_b + gamma * next_q_values

                # 4.2 计算当前 Q(s, a) 并最小化 MSE
                current_q = critic(states_b, actions_b)
                critic_loss = nn.MSELoss()(current_q, y)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # ---- Actor 更新 ----
                # 4.3 最大化 Q(s, π(s)) => 最小化 -Q(...)
                current_actions = actor(states_b)
                actor_loss = - critic(states_b, current_actions).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ---- 软更新目标网络 ----
                soft_update(target_actor, actor, tau=0.005)
                soft_update(target_critic, critic, tau=0.005)

        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")

    plt.ioff()
    return rewards_history

# ========== 7. 主程序入口 ==========
if __name__ == "__main__":
    env = HysteresisEnv()
    # state_dim=2 (param1,param2), action_dim=2
    actor = Actor(state_dim=2, action_dim=2)
    critic = Critic(state_dim=2, action_dim=2)

    actor_optimizer = optim.AdamW(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=3e-4)
    buffer = ReplayBuffer(capacity=50000)

    # 可选：如果已有模型可加载
    # try:
    #     checkpoint = torch.load("model.pth")
    #     actor.load_state_dict(checkpoint['actor'])
    #     critic.load_state_dict(checkpoint['critic'])
    #     actor_optimizer.load_state_dict(checkpoint['actor_optim'])
    #     critic_optimizer.load_state_dict(checkpoint['critic_optim'])
    #     print("Loaded existing model")
    # except:
    #     print("Training new model from scratch")

    rewards = train_ddpg(env, actor, critic, actor_optimizer, critic_optimizer,
                         buffer,
                         num_episodes=50,   # 可自行调节
                         max_steps=1000,     # 每个episode的step数
                         batch_size=64,
                         gamma=0.99)

    # 训练完可保存模型
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'actor_optim': actor_optimizer.state_dict(),
        'critic_optim': critic_optimizer.state_dict()
    }, "model_final.pth")

    final_params = env.current_params
    pd.DataFrame({'Episodes': range(len(rewards)), 'Rewards': rewards}).to_csv('training_history.csv', index=False)
    print(f"Final parameters: {final_params}")
    print(f"Target parameters: {env.target_params}")
