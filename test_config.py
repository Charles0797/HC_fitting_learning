# 保存为test_config.py
def get_test_config():
    """首次测试用的小规模配置"""
    return {
        'num_episodes': 50,           # 很少的episode数
        'max_steps': 10,              # 很少的步数
        'batch_size': 16,             # 小批量
        'buffer_size': 500,           # 小buffer
        'n_parallel': 1,              # 单候选
        'num_workers': 1,             # 单进程
        'use_multiprocessing': False, # 禁用多进程
        'save_every': 10,             # 频繁保存
        'gamma': 0.98,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'tau': 0.005,
        'buffer_save_path': 'test_buffer.pkl.gz',
        'model_path': 'test_model.pth'
    }