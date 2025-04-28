import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import openseespy.opensees as ops
import pickle
import os

# ========== 1. ScalingLayer：可学习的输入缩放层 ==========
class ScalingLayer(nn.Module):
    def __init__(self, input_dim):
        super(ScalingLayer, self).__init__()
        # 初始 log_scale = 0 => exp(0) = 1
        self.log_scale = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # 保证缩放因子始终为正
        return x * torch.exp(self.log_scale)


# ========== 2. OU噪声 ==========
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def noise(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# ========== 3. 环境：HysteresisEnv ==========
class HysteresisEnv:
    """
    环境基于参数 p1, p2 的识别任务。
    target_params = [300, 200, 0.1] 中的前两个维度用于与环境交互，
    第三个 (0.1) 仅在 hysteretic_curve() 中使用。
    """

    def __init__(self):
        # 两个可调参数的取值范围
        self.param_ranges = [[50, 400], [50, 400]]
        # 从 Excel 文件中读取数据
        df = pd.read_excel("3圈steel01.xlsx")
        self.protocol = df["displacement"].values
        self.target_curve = df["force"].values

        if self.protocol[0] != 0 or self.target_curve[0] != 0:
            print("警告：目标曲线起始点不为 (0,0)，请检查数据！")
        self.target_range = np.max(self.target_curve) - np.min(self.target_curve)
        self.reset()

    def random_params(self):
        return [random.uniform(*r) for r in self.param_ranges]

    def normalize_state(self, params):
        """归一化到[-1, 1]，公式： (p - mid) / (half_range) """
        norm_params = []
        for p, r in zip(params, self.param_ranges):
            lower, upper = r
            mid = (lower + upper) / 2.0
            half_range = (upper - lower) / 2.0
            norm_params.append((p - mid) / half_range)
        return np.array(norm_params)

    def hysteretic_curve(self, params):
        """
        建立 OpenSees 模型，施加 self.protocol 位移，返回力-位移曲线（2000个数据点）。
        """
        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)
        ops.node(1, 0.0)
        ops.node(2, 0.0)
        ops.fix(1, 1)
        # p1= params[0], p2= params[1], strainHardeningRatio= 0.1 固定
        ops.uniaxialMaterial("Steel01", 1, params[0], params[1], 0.1)
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
                result.append(0.0)
        ops.wipe()
        return np.array(result)

    def step(self, action):
        """
        执行动作，返回 (next_state, reward, done)
        action: [a1, a2]（对 current_params 做加减操作）
        1）更新参数，若更新后参数正好在边界则结束该回合；
        2）计算拟合曲线与目标曲线的 L2 范数 error_norm，
           根据 error_norm 给定奖励：
             error_norm > 1000  → reward = 0
             100 < error_norm ≤ 1000 → reward = 1
             10 < error_norm ≤ 100 → reward = 2
             0 ≤ error_norm ≤ 10  → reward = 3
        """
        scale_factors = [1, 1]  # 可根据需要设置更新尺度
        new_params = []
        done = False
        for i, (p, a, r) in enumerate(zip(self.current_params, action, self.param_ranges)):
            new_val = p + scale_factors[i] * a
            # clip 到 [lower, upper]
            lower, upper = r
            new_val = max(min(new_val, upper), lower)
            new_params.append(new_val)
            # 当参数等于边界时，则终止该回合
            if new_val == lower or new_val == upper:
                done = True
        self.current_params = new_params

        # 计算拟合曲线及误差
        fitted_curve = self.hysteretic_curve(self.current_params)
        error_norm = np.linalg.norm(self.target_curve - fitted_curve, ord=2)
        # if error_norm > 1000:
        #     base_reward = 0.01
        # elif error_norm > 500:
        #     base_reward = 1
        # elif error_norm > 100:
        #     base_reward = 10
        # else:
        #     base_reward = 30
        #
        # reward = base_reward

        reward = (10/(np.exp(error_norm/1000))**2)

        # 返回归一化后的状态
        norm_state = self.normalize_state(self.current_params)
        return norm_state, reward, done

    def reset(self):
        """
        重置环境状态，随机初始化 current_params 并返回归一化状态
        """
        self.current_params = self.random_params()
        return self.normalize_state(self.current_params)


# ========== 4. Actor 与 Critic 网络 ==========
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # 注意：环境返回的状态已归一化，不需要在这里再做归一化处理
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
        sa = torch.cat([state, action], dim=1)
        sa = self.scaling(sa)
        q = self.fc(sa)
        return q


# ========== 5. ReplayBuffer ==========
class ReplayBuffer:
    def __init__(self, capacity, filename='replay_buffer.pkl'):
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
        states, actions, rewards, next_states = [], [], [], []
        for s, a, r, s_ in batch:
            states.append(s)
            actions.append(a)
            rewards.append([r])
            next_states.append(s_)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

    # 保存当前的 buffer 到文件
    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved to {self.filename}")

    # 从文件中加载 buffer
    def load(self):
        try:
            with open(self.filename, 'rb') as f:
                self.buffer = pickle.load(f)
            print(f"Replay buffer loaded from {self.filename}")
        except FileNotFoundError:
            print(f"No saved buffer found at {self.filename}, starting with empty buffer.")
            self.buffer = []



# ========== 6. DDPG 软更新 ==========
def soft_update(target_net, source_net, tau=0.01):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


# ========== 7. 训练函数：包含 DDPG 更新逻辑 ==========
def train_ddpg(env, actor, critic, actor_optimizer, critic_optimizer,
               buffer, num_episodes=100, max_steps=500, batch_size=64, gamma=0.98):
    # 构建目标网络并拷贝参数
    target_actor = Actor(2, 2)
    target_critic = Critic(2, 2)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    target_actor.eval()
    target_critic.eval()

    rewards_history = []
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    target_curve = env.target_curve
    target_min = np.min(target_curve)
    target_max = np.max(target_curve)
    fixed_xlim = (-7.5, 7.5)
    fixed_ylim = (target_min - 50, target_max + 50)

    # 初始化 OU 噪声
    ou_noise = OUNoise(action_dim=2, mu=0.0, theta=0.15, sigma=0.2)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_rewards = []
        ou_noise.reset()  # 每个 episode 重置 OU 噪声

        for step in range(max_steps):
            # 1. 从 Actor 网络得到动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            # 2. 添加 OU 噪声（探索）
            action += ou_noise.noise()
            # 将动作 clip 到 [-1,1]
            action = np.clip(action, -1.0, 1.0)

            # 3. 与环境交互
            next_state, reward, done = env.step(action)
            buffer.add((state, action, reward, next_state))

            episode_reward += reward
            step_rewards.append(reward)
            state = next_state

            # print(f"Episode {episode + 1} Step {step}: Params (raw)={env.current_params}")

            # 可视化
            if step % 10 == 0:
                fitted_curve = env.hysteretic_curve(env.current_params)
                ax1.clear()
                ax1.plot(env.protocol, target_curve, 'b-', label='Target')
                ax1.plot(env.protocol, fitted_curve, 'r--', label='Current')
                param_info = "/".join([f"{p:.2f}" for p in env.current_params])
                ax1.set_title(f'Episode {episode + 1} Step {step}\nParams: {param_info}')
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

            # 如果达到终止条件，退出当前 episode（参数处于边界）
            if done:
                print("参数达到边界，结束当前回合。")
                break

            # 如果累计回合奖励大于500，结束该回合
            if episode_reward > 10000:
                print("累计回合奖励大于10000，结束当前回合。")
                break

            # 4. 若 buffer 中数据足够，则进行一次 DDPG 更新
            if len(buffer) > batch_size:
                states_b, actions_b, rewards_b, next_states_b = buffer.sample(batch_size)
                # ---- Critic 更新 ----
                with torch.no_grad():
                    next_actions = target_actor(next_states_b)
                    next_q_values = target_critic(next_states_b, next_actions)
                    y = rewards_b + gamma * next_q_values
                current_q = critic(states_b, actions_b)
                critic_loss = nn.MSELoss()(current_q, y)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=100.0)

                # ---- Actor 更新 ----
                current_actions = actor(states_b)
                actor_loss = - critic(states_b, current_actions).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ---- 软更新目标网络 ----
                soft_update(target_actor, actor, tau=0.01)
                soft_update(target_critic, critic, tau=0.01)

        rewards_history.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Episode Reward: {episode_reward:.2f}")

    plt.ioff()
    return rewards_history


# ========== 8. 主程序入口 ==========
if __name__ == "__main__":
    env = HysteresisEnv()
    # 状态维度=2（归一化后的参数），动作维度=2
    actor = Actor(state_dim=2, action_dim=2)
    critic = Critic(state_dim=2, action_dim=2)

    actor_optimizer = optim.AdamW(actor.parameters(), lr=1e-5)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=1e-5)

    # 尝试加载已保存的模型
    model_path = "model_final.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        actor_optimizer.load_state_dict(checkpoint['actor_optim'])
        critic_optimizer.load_state_dict(checkpoint['critic_optim'])
        print("已加载保存的模型，并继续训练。")
    else:
        print("未找到保存的模型，从头开始训练。")

    # 初始化 replay buffer，若有保存的 buffer 也可类似加载
    buffer = ReplayBuffer(capacity=50000, filename='replay_buffer.pkl')
    buffer.load()

    rewards = train_ddpg(env, actor, critic, actor_optimizer, critic_optimizer,
                         buffer,
                         num_episodes=100,  # 可调整
                         max_steps=500,  # 每个 episode 最大步数
                         batch_size=64,
                         gamma=0.98)

    # 训练完成后保存模型
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'actor_optim': actor_optimizer.state_dict(),
        'critic_optim': critic_optimizer.state_dict()
    }, model_path)

    # 保存 replay buffer（如果需要连续使用之前的数据）
    buffer.save()

    final_params = env.current_params
    pd.DataFrame({'Episodes': range(len(rewards)), 'Rewards': rewards}).to_csv('training_history.csv', index=False)
    print(f"Final parameters (raw): {final_params}")
    print("Target parameters: 300, 200, 0.1")

