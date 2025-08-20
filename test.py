# 首先搜索框搜索config查看配置是否需要更改
# 再搜索pth查看训练模型是否配置一致
# 搜索框搜索steel01data查看文件是否下载，位置是否正确
# 搜索框搜索（250,15000,0.1）3圈steel01下载位置是否正确
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Steel01 参数识别 — 优化版本（适配8核16GB配置）
主要优化：
- Windows多进程兼容性修复
- 内存使用优化（16GB友好）
- OpenSees稳定性增强
- 系统资源监控
- 错误处理改进
"""

import os
import sys
import tempfile
import shutil
import gzip
import pickle
import logging
import gc
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from vacation_task.improve.test_config import get_test_config

# 尝试导入psutil用于系统监控（如果没有可以pip install psutil）
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: 未安装psutil，无法进行系统资源监控。建议运行: pip install psutil")

# ------------------- 日志配置 -------------------
LOGFILE = "training_log.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOGFILE, encoding='utf-8')])
logger = logging.getLogger(__name__)


# ------------------- 系统资源监控 -------------------
def monitor_system_resources():
    """
    监控系统资源使用情况
    返回系统CPU、内存使用情况的字典
    """
    if not PSUTIL_AVAILABLE:
        return None

    try:
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024 ** 3)  # GB

        # 当前进程内存使用情况
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 ** 3)  # GB

        logger.info(f"系统监控 - CPU: {cpu_percent:.1f}%, "
                    f"内存: {memory_percent:.1f}% (可用: {memory_available:.1f}GB), "
                    f"进程内存: {process_memory:.2f}GB")

        # 内存使用警告
        if memory_percent > 85:
            logger.warning("⚠️  内存使用率过高！建议减少batch_size或buffer_size")

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_gb': memory_available,
            'process_memory_gb': process_memory
        }
    except Exception as e:
        logger.warning(f"无法获取系统资源信息: {e}")
        return None


def force_garbage_collection():
    """
    强制进行垃圾回收，释放内存
    """
    collected = gc.collect()
    logger.info(f"垃圾回收完成，释放了 {collected} 个对象")


# ------------------- 数据加载 -------------------
def load_training_data_from_txt(txt_file_path):
    """
    从txt文件加载训练数据

    参数:
        txt_file_path: txt文件路径

    返回:
        包含训练数据的列表，每个元素包含params、protocol、curve、filename
    """
    logger.info(f"正在加载训练数据: {txt_file_path} (新格式)")

    if not os.path.exists(txt_file_path):
        logger.error(f"文件不存在: {txt_file_path}")
        return []

    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return []

    if len(lines) < 12010:
        logger.error(f"错误: 文件行数不足 ({len(lines)}行)，需要至少12010行")
        return []

    num_columns = len(lines[0].split('\t'))
    all_data = []
    logger.info(f"检测到 {num_columns} 个样本列")

    # 逐列解析数据
    for col_idx in range(num_columns):
        try:
            # 读取参数（前3行）
            fy = float(lines[0].split('\t')[col_idx])
            E = float(lines[1].split('\t')[col_idx])
            b = float(lines[2].split('\t')[col_idx])

            # 读取位移数据（第10-6009行，共6000个点）
            displacement = []
            for row_idx in range(10, 6010):
                if row_idx < len(lines):
                    parts = lines[row_idx].split('\t')
                    if col_idx < len(parts) and parts[col_idx].strip():
                        displacement.append(float(parts[col_idx]))

            # 读取力数据（第6010-12009行，共6000个点）
            force = []
            for row_idx in range(6010, 12010):
                if row_idx < len(lines):
                    parts = lines[row_idx].split('\t')
                    if col_idx < len(parts) and parts[col_idx].strip():
                        force.append(float(parts[col_idx]))

            # 数据长度验证
            if len(displacement) != 6000 or len(force) != 6000:
                logger.warning(f"样本 {col_idx + 1} 数据长度不匹配 - 位移: {len(displacement)} 点, 力: {len(force)} 点")
                if len(displacement) > 6000:
                    displacement = displacement[:6000]
                if len(force) > 6000:
                    force = force[:6000]
                if len(displacement) < 6000 or len(force) < 6000:
                    continue

            # 添加到数据列表
            all_data.append({
                'params': [fy, E, b],
                'protocol': np.array(displacement, dtype=np.float32),
                'curve': np.array(force, dtype=np.float32),
                'filename': f'data_{col_idx + 1}'
            })

            # 进度提示
            if (col_idx + 1) % 10 == 0:
                logger.info(f"已加载样本 {col_idx + 1}/{num_columns}")

        except Exception as e:
            logger.exception(f"加载样本 {col_idx + 1} 时出错: {e}")
            continue

    logger.info(f"总共加载了 {len(all_data)} 组训练数据")

    # 打印参数范围统计
    if all_data:
        params_array = np.array([d['params'] for d in all_data])
        logger.info("参数范围统计：")
        names = ['fy', 'E', 'b']
        for i, n in enumerate(names):
            logger.info(
                f"{n}: min={params_array[:, i].min():.2f}, max={params_array[:, i].max():.2f}, mean={params_array[:, i].mean():.2f}")

    return all_data


def load_test_data_from_excel(excel_file_path):
    """
    从Excel文件加载测试数据

    参数:
        excel_file_path: Excel文件路径

    返回:
        tuple: (displacement, force) 数组
    """
    if not os.path.exists(excel_file_path):
        logger.error(f"Excel文件不存在: {excel_file_path}")
        return np.array([]), np.array([])

    try:
        df = pd.read_excel(excel_file_path)

        # 尝试不同的列名组合
        if 'displacement' in df.columns and 'force' in df.columns:
            displacement = df['displacement'].values
            force = df['force'].values
        elif '位移' in df.columns and '力' in df.columns:
            displacement = df['位移'].values
            force = df['力'].values
        else:
            # 使用前两列
            displacement = df.iloc[:, 0].values
            force = df.iloc[:, 1].values

        logger.info(f"成功加载测试数据: {len(displacement)} 个数据点")
        return displacement.astype(np.float32), force.astype(np.float32)

    except Exception as e:
        logger.error(f"加载Excel文件失败: {e}")
        return np.array([]), np.array([])


# ------------------- 改进的多进程Worker -------------------
def worker_hysteretic_batch_improved(args):
    """
    改进的多进程worker，用于并行计算滞回曲线
    增强了错误处理和稳定性

    参数:
        args: tuple (params_list, protocol)

    返回:
        np.array: shape (len(params_list), n_steps) 的力响应数组
    """
    params_list, protocol = args

    # 进程级别的异常处理
    try:
        # 重新导入opensees（避免进程间冲突）
        import openseespy.opensees as ops

        # 确保每个worker进程中opensees是干净的
        try:
            ops.wipe()
        except:
            pass

    except ImportError as e:
        print(f"Worker进程无法导入opensees: {e}")
        return np.zeros((len(params_list), len(protocol)), dtype=float)
    except Exception as e:
        print(f"Worker初始化失败: {e}")
        return np.zeros((len(params_list), len(protocol)), dtype=float)

    protocol = np.asarray(protocol, dtype=np.float64)
    n_steps = len(protocol)
    n_models = len(params_list)
    results = np.zeros((n_models, n_steps), dtype=float)

    try:
        # OpenSees模型设置
        ops.model('basic', '-ndm', 1, '-ndf', 1)
        ops.node(1, 0.0)
        ops.node(2, 0.0)
        ops.fix(1, 1)

        # 创建材料和单元（支持多个参数组合）
        valid_elements = []
        for i, p in enumerate(params_list, start=1):
            try:
                fy, E, b = p
                # 参数有效性验证
                if fy <= 0 or E <= 0 or b <= 0 or fy > 10000 or E > 100000 or b > 1:
                    print(f"参数无效: fy={fy}, E={E}, b={b}")
                    continue

                mat_id = i
                ele_id = i
                ops.uniaxialMaterial('Steel01', mat_id, float(fy), float(E), float(b))
                ops.element('twoNodeLink', ele_id, 1, 2, '-mat', mat_id, '-dir', 1)
                valid_elements.append(i)

            except Exception as mat_e:
                print(f"创建材料/单元 {i} 失败: {mat_e}")
                continue

        if not valid_elements:
            print("没有有效的单元被创建")
            ops.wipe()
            return results

        # 分析设置
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(2, 1.0)
        ops.algorithm('Newton')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.system('BandSPD')
        ops.integrator("DisplacementControl", 2, 1, 0.0)
        ops.analysis('Static')

        # 逐步分析
        prev_disp = 0.0
        consecutive_failures = 0
        max_consecutive_failures = 10  # 允许的连续失败次数

        for step_idx, disp in enumerate(protocol):
            delta_disp = float(disp - prev_disp)
            prev_disp = disp

            try:
                ops.integrator('DisplacementControl', 2, 1, delta_disp)
                ok = ops.analyze(1)

                if ok == 0:
                    consecutive_failures = 0  # 重置失败计数
                    # 提取每个单元的力
                    for i in valid_elements:
                        try:
                            fvec = ops.eleForce(i)
                            if isinstance(fvec, (list, tuple, np.ndarray)):
                                val = float(fvec[0]) if len(fvec) > 0 else 0.0
                            else:
                                val = float(fvec)
                            # 检查数值有效性
                            if np.isnan(val) or np.isinf(val):
                                val = 0.0
                        except Exception:
                            val = 0.0
                        results[i - 1, step_idx] = val
                else:
                    consecutive_failures += 1
                    # 连续失败太多次则提前终止
                    if consecutive_failures > max_consecutive_failures:
                        print(f"连续失败次数超过 {max_consecutive_failures}，提前终止分析")
                        break
                    # 设置当前步的力为0
                    for i in valid_elements:
                        results[i - 1, step_idx] = 0.0

            except Exception as step_e:
                print(f"分析步骤 {step_idx} 失败: {step_e}")
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    break

        ops.wipe()

    except Exception as e:
        print(f"Worker OpenSees分析失败: {e}")
        try:
            ops.wipe()
        except:
            pass

    return results


# ------------------- 优化的多目标滞回环境 -------------------
class MultiTargetHysteresisEnvOptimized:
    """
    优化版的多目标滞回环境类
    - 减少了特征维度以降低内存使用
    - 添加了结果缓存机制
    - 改进了错误处理
    """

    def __init__(self, target_data, feature_dim=30):  # 从50减少到30
        # Steel01材料参数范围
        self.param_ranges = [
            [1, 1000],  # fy: 屈服强度
            [1, 50000],  # E: 弹性模量
            [0.01, 0.3]  # b: 应变硬化比
        ]
        # 参数调整的比例因子
        self.scale_factors = [10, 500, 0.005]

        self.target_data = target_data
        self.num_targets = len(target_data) if target_data else 0
        self.feature_dim = feature_dim

        self.current_target_idx = None
        self.current_target = None
        self.current_params = None

        # 统计信息
        self.episode_count = 0
        self.success_count = 0
        self.opensees_failure_count = 0
        self.opensees_total_count = 0

        # 结果缓存（减少重复计算）
        self._curve_cache = {}
        self.max_cache_size = 500  # 限制缓存大小避免内存溢出

    def reset(self, target_idx=None):
        """重置环境到新的episode"""
        self.episode_count += 1

        if self.num_targets == 0:
            self.current_params = self.random_params()
            return self._get_state()

        # 选择目标
        if target_idx is None:
            self.current_target_idx = np.random.randint(self.num_targets)
        else:
            self.current_target_idx = target_idx

        self.current_target = self.target_data[self.current_target_idx]
        self.current_params = self.random_params()

        return self._get_state()

    def random_params(self):
        """生成随机的材料参数"""
        return [np.random.uniform(low, high) for low, high in self.param_ranges]

    def normalize_params(self, params):
        """将参数归一化到[-1, 1]范围"""
        norm = []
        for p, (low, high) in zip(params, self.param_ranges):
            norm_p = 2 * (p - low) / (high - low) - 1
            norm.append(np.clip(norm_p, -1, 1))  # 确保在范围内
        return np.array(norm, dtype=np.float32)

    def _get_state(self):
        """获取当前状态（特征向量）"""
        norm_params = self.normalize_params(self.current_params)

        if self.current_target is None:
            return np.concatenate([norm_params, np.zeros(self.feature_dim + 10, dtype=np.float32)]).astype(np.float32)

        # 提取目标曲线特征
        target_features = self._extract_curve_features_optimized(
            self.current_target['curve'],
            self.current_target['protocol']
        )

        # 计算当前参数对应的曲线
        current_curve = self.hysteretic_curve_single(self.current_params)

        # 计算误差特征
        error_features = self._compute_error_features(current_curve, self.current_target['curve'])

        # 组合所有特征
        state = np.concatenate([norm_params, target_features, error_features])
        return state.astype(np.float32)

    def _extract_curve_features_optimized(self, curve, protocol):
        """
        优化的曲线特征提取（减少计算量和特征维度）
        """
        # 生成缓存键
        cache_key = hash((tuple(curve[:100]), tuple(protocol[:100])))  # 只用前100个点计算hash
        if cache_key in self._curve_cache:
            return self._curve_cache[cache_key]

        features = []

        # 基本统计特征 (5个)
        features.extend([
            np.max(curve), np.min(curve),
            np.mean(curve), np.std(curve),
            np.max(curve) - np.min(curve)
        ])

        # 零交叉点数量 (1个)
        zero_crossings = np.where(np.diff(np.sign(curve)))[0]
        features.append(len(zero_crossings))

        # 峰值数量 (1个)
        try:
            peaks, _ = find_peaks(np.abs(curve))
            features.append(len(peaks))
        except Exception:
            features.append(0)

        # 能量特征 (1个)
        try:
            energy = np.trapz(np.abs(curve), np.abs(protocol))
            features.append(energy)
        except Exception:
            features.append(0)

        # 初始刚度 (1个)
        n_initial = min(max(len(curve) // 20, 5), len(curve))
        if n_initial >= 2:
            try:
                initial_slope, _ = np.polyfit(protocol[:n_initial], curve[:n_initial], 1)
                features.append(initial_slope)
            except:
                features.append(0)
        else:
            features.append(0)

        # 简化的采样特征 (12个，从20减少到12)
        n_samples = 12
        if len(curve) >= n_samples:
            indices = np.linspace(0, len(curve) - 1, n_samples, dtype=int)
            sampled = curve[indices]
            curve_range = np.max(np.abs(curve)) + 1e-6
            features.extend((sampled / curve_range).tolist())
        else:
            features.extend(curve.tolist())
            features.extend([0] * (n_samples - len(curve)))

        # 转换为numpy数组并调整长度
        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]

        # 缓存结果（控制缓存大小）
        if len(self._curve_cache) < self.max_cache_size:
            self._curve_cache[cache_key] = features
        elif len(self._curve_cache) >= self.max_cache_size:
            # 清理一半缓存
            keys_to_remove = list(self._curve_cache.keys())[:self.max_cache_size // 2]
            for k in keys_to_remove:
                del self._curve_cache[k]
            self._curve_cache[cache_key] = features

        return features

    def _compute_error_features(self, current_curve, target_curve):
        """计算当前曲线与目标曲线的误差特征"""
        min_len = min(len(current_curve), len(target_curve))
        if min_len == 0:
            return np.zeros(10, dtype=np.float32)

        current_curve = current_curve[:min_len]
        target_curve = target_curve[:min_len]
        diff = current_curve - target_curve
        abs_diff = np.abs(diff)

        features = []

        # 基本误差统计 (4个)
        features.extend([
            np.mean(abs_diff),
            np.max(abs_diff),
            np.sqrt(np.mean(diff ** 2)),  # RMSE
            np.std(diff)
        ])

        # 符号统计 (2个)
        features.extend([
            np.sum(diff > 0) / (len(diff) + 1),
            np.sum(diff < 0) / (len(diff) + 1)
        ])

        # 分段误差 (4个)
        n_segments = 4
        segment_size = max(1, len(diff) // n_segments)
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(diff)
            seg_err = np.mean(abs_diff[start_idx:end_idx]) if end_idx > start_idx else 0.0
            features.append(seg_err)

        return np.array(features, dtype=np.float32)

    def hysteretic_curve_single(self, params, protocol=None):
        """单个参数组合的滞回曲线计算（包装batch版本）"""
        res = self.hysteretic_curve_batch([params], protocol=protocol)
        return res[0] if len(res) > 0 else np.array([])

    def hysteretic_curve_batch(self, params_list, protocol=None):
        """
        批量计算滞回曲线（单进程版本）
        使用OpenSees的twoNodeLink单元批量计算多个参数组合

        参数:
            params_list: 参数列表，每个元素为[fy, E, b]
            protocol: 位移历程，如果为None则使用当前目标的protocol

        返回:
            np.array: shape (n_models, n_steps) 的力响应数组
        """
        try:
            import openseespy.opensees as ops
        except Exception as e:
            logger.exception("无法导入 openseespy: %s", e)
            # 如果OpenSees不可用，返回零数组
            protocol = np.asarray(protocol) if protocol is not None else (
                self.current_target['protocol'] if self.current_target else np.zeros(100)
            )
            n_steps = len(protocol)
            n_models = len(params_list)
            return np.zeros((n_models, n_steps), dtype=float)

        if protocol is None:
            protocol = self.current_target['protocol'] if self.current_target is not None else np.zeros(100)

        protocol = np.asarray(protocol, dtype=np.float64)
        n_steps = len(protocol)
        n_models = len(params_list)
        results = np.zeros((n_models, n_steps), dtype=float)

        try:
            # 清理并建立新模型
            ops.wipe()
            ops.model('basic', '-ndm', 1, '-ndf', 1)
            ops.node(1, 0.0)
            ops.node(2, 0.0)
            ops.fix(1, 1)

            # 为每个参数组合创建材料和单元
            valid_elements = []
            for i, p in enumerate(params_list, start=1):
                try:
                    fy, E, b = p

                    # 参数有效性检查
                    if not (0 < fy < 10000 and 0 < E < 100000 and 0 < b < 1):
                        logger.warning(f"参数超出合理范围: fy={fy}, E={E}, b={b}")
                        continue

                    mat_id = i
                    ele_id = i
                    ops.uniaxialMaterial('Steel01', mat_id, float(fy), float(E), float(b))
                    ops.element('twoNodeLink', ele_id, 1, 2, '-mat', mat_id, '-dir', 1)
                    valid_elements.append(i)

                except Exception as e:
                    logger.warning(f"创建材料/单元 {i} 失败: {e}")
                    continue

            if not valid_elements:
                logger.warning("没有有效的单元被创建")
                ops.wipe()
                return results

            # 设置分析
            ops.timeSeries('Linear', 1)
            ops.pattern('Plain', 1, 1)
            ops.load(2, 1.0)
            ops.algorithm('Newton')
            ops.numberer('RCM')
            ops.constraints('Transformation')
            ops.system('BandSPD')
            ops.integrator("DisplacementControl", 2, 1, 0.0)
            ops.analysis('Static')

            # 逐步加载
            prev_disp = 0.0
            for step_idx, disp in enumerate(protocol):
                delta_disp = float(disp - prev_disp)
                prev_disp = disp

                try:
                    ops.integrator('DisplacementControl', 2, 1, delta_disp)
                    ok = ops.analyze(1)
                    self.opensees_total_count += 1

                    if ok == 0:
                        # 成功分析，提取每个单元的力
                        for i in valid_elements:
                            try:
                                fvec = ops.eleForce(i)
                                if isinstance(fvec, (list, tuple, np.ndarray)):
                                    if len(fvec) == 0:
                                        val = 0.0
                                    else:
                                        val = float(fvec[0])
                                else:
                                    val = float(fvec)

                                # 检查数值有效性
                                if np.isnan(val) or np.isinf(val):
                                    val = 0.0

                            except Exception:
                                val = 0.0

                            results[i - 1, step_idx] = val
                    else:
                        # 分析失败，设置力为0
                        self.opensees_failure_count += 1
                        for i in valid_elements:
                            results[i - 1, step_idx] = 0.0

                except Exception as e:
                    logger.warning(f"步骤 {step_idx} 分析失败: {e}")
                    self.opensees_failure_count += 1
                    for i in valid_elements:
                        results[i - 1, step_idx] = 0.0

            ops.wipe()

        except Exception as e:
            logger.exception("[hysteretic_curve_batch] OpenSees 分析出错: %s", e)
            try:
                ops.wipe()
            except Exception:
                pass
            return np.zeros((n_models, n_steps), dtype=float)

        return results

    # 兼容性接口
    def hysteretic_curve(self, params, protocol=None):
        """兼容性接口，调用单个曲线计算"""
        return self.hysteretic_curve_single(params, protocol=protocol)

    def step(self, action):
        """
        执行一个动作步骤

        参数:
            action: 动作向量 [delta_fy, delta_E, delta_b]

        返回:
            tuple: (next_state, reward, done, info)
        """
        # 根据动作更新参数
        new_params = []
        hit_boundary = False

        for i, (p, a) in enumerate(zip(self.current_params, action)):
            delta = self.scale_factors[i] * a
            new_p = p + delta
            low, high = self.param_ranges[i]

            if new_p <= low:
                new_p = low
                hit_boundary = True
            elif new_p >= high:
                new_p = high
                hit_boundary = True

            new_params.append(new_p)

        self.current_params = new_params

        # 如果没有目标，返回默认值
        if self.current_target is None:
            next_state = self._get_state()
            return next_state, 0, False, {}

        # 计算当前参数对应的曲线
        fitted_curve = self.hysteretic_curve(self.current_params)
        target_curve = self.current_target['curve']

        # 计算误差
        min_len = min(len(fitted_curve), len(target_curve))
        if min_len == 0:
            error_norm = float('inf')
            relative_error = 1.0
        else:
            fitted_curve = fitted_curve[:min_len]
            target_curve = target_curve[:min_len]
            error_norm = np.linalg.norm(fitted_curve - target_curve)
            relative_error = error_norm / (np.linalg.norm(target_curve) + 1e-6)

        # 计算奖励
        # 在step方法中修改奖励计算：
        base_reward = np.exp(10 / relative_error)  # 更敏感
        # 精度奖励
        if relative_error < 0.01:
            precision_bonus = 10.0
        elif relative_error < 0.05:
            precision_bonus = 5.0
        elif relative_error < 0.1:
            precision_bonus = 1.0
        else:
            precision_bonus = 0.0

        reward = base_reward + precision_bonus - 0.01
        done = True
        if hit_boundary:
            done = True
            reward -= 1.0

        if relative_error < 0.05:
            done = True
            reward += 50.0
            self.success_count += 1

        next_state = self._get_state()

        info = {
            'error_norm': error_norm,
            'relative_error': relative_error,
            'current_params': self.current_params.copy(),
            'target_params': self.current_target['params'] if self.current_target else None,
            'target_idx': self.current_target_idx,
            'hit_boundary': hit_boundary
        }

        return next_state, reward, done, info


# ------------------- 改进的优先级经验回放缓冲区 -------------------
class PrioritizedReplayBufferOptimized:
    """
    优化版的优先级经验回放缓冲区
    - 改进了内存管理
    - 增强了数据验证
    - 优化了序列化性能
    """

    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.target_counts = {}

    def add(self, experience, error=None):
        """
        添加经验到缓冲区

        参数:
            experience: (state, action, reward, next_state, done, info)
            error: TD误差，用于计算优先级
        """
        # 标准化经验数据类型
        state, action, reward, next_state, done, info = experience

        # 确保数据类型一致性
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = float(reward)
        done = bool(done)

        # 数据有效性检查
        if np.any(np.isnan(state)) or np.any(np.isnan(action)) or np.any(np.isnan(next_state)):
            logger.warning("检测到NaN值，跳过添加到buffer")
            return

        exp = (state, action, reward, next_state, done, info)

        # 更新目标计数
        target_idx = info.get('target_idx', None) if isinstance(info, dict) else None
        if target_idx is not None:
            self.target_counts[target_idx] = self.target_counts.get(target_idx, 0) + 1

        # 计算优先级
        if error is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        else:
            priority = (abs(float(error)) + 1e-6) ** self.alpha

        # 添加或替换经验
        if self.size < self.capacity:
            self.buffer.append(exp)
        else:
            # 更新目标计数（移除旧经验的计数）
            old_info = self.buffer[self.position][5]
            old_tidx = old_info.get('target_idx', None) if isinstance(old_info, dict) else None
            if old_tidx is not None and old_tidx in self.target_counts:
                self.target_counts[old_tidx] = max(0, self.target_counts[old_tidx] - 1)
            self.buffer[self.position] = exp

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        采样一批经验

        参数:
            batch_size: 批量大小

        返回:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if self.size == 0:
            return None

        # 获取优先级并计算采样概率
        priorities = self.priorities[:self.size].copy()
        pri_sum = priorities.sum()

        if pri_sum <= 0 or np.isnan(pri_sum):
            # 如果优先级无效，使用均匀分布
            probs = np.ones(self.size) / self.size
        else:
            probs = priorities / pri_sum

        # 采样索引
        try:
            indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        except Exception as e:
            logger.warning(f"采样失败，使用随机采样: {e}")
            indices = np.random.choice(self.size, batch_size, replace=True)
            probs = np.ones(self.size) / self.size

        # 提取经验
        experiences = [self.buffer[idx] for idx in indices]

        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        if np.max(weights) == 0:
            weights = np.ones_like(weights)
        else:
            weights = weights / weights.max()

        # 转换为张量
        try:
            states_np = np.array([e[0] for e in experiences], dtype=np.float32)
            actions_np = np.array([e[1] for e in experiences], dtype=np.float32)
            rewards_np = np.array([[e[2]] for e in experiences], dtype=np.float32)
            next_states_np = np.array([e[3] for e in experiences], dtype=np.float32)
            dones_np = np.array([[1.0 if e[4] else 0.0] for e in experiences], dtype=np.float32)

            states = torch.from_numpy(states_np)
            actions = torch.from_numpy(actions_np)
            rewards = torch.from_numpy(rewards_np)
            next_states = torch.from_numpy(next_states_np)
            dones = torch.from_numpy(dones_np)
            weights = torch.FloatTensor(weights).unsqueeze(1)
        except Exception as e:
            logger.error(f"张量转换失败: {e}")
            return None

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        """更新指定索引的优先级"""
        for idx, err in zip(indices, errors):
            if 0 <= idx < self.size:
                self.priorities[idx] = (abs(float(err)) + 1e-6) ** self.alpha

    def to_serializable(self):
        """转换为可序列化的格式"""
        serial = {
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'position': self.position,
            'size': self.size,
            'priorities': self.priorities[:self.size].tolist(),
            'buffer': [],
            'target_counts': self.target_counts
        }

        # 序列化buffer内容
        for exp in self.buffer[:self.size]:
            state, action, reward, next_state, done, info = exp
            serial['buffer'].append((
                state.tolist(),
                action.tolist(),
                reward,
                next_state.tolist(),
                bool(done),
                info
            ))

        return serial

    @classmethod
    def from_serializable(cls, serial):
        """从序列化格式恢复对象"""
        obj = cls(
            capacity=serial.get('capacity', 50000),
            alpha=serial.get('alpha', 0.6),
            beta=serial.get('beta', 0.4),
            beta_increment=serial.get('beta_increment', 0.001)
        )

        obj.position = serial.get('position', 0)
        obj.size = serial.get('size', 0)

        # 恢复优先级
        priorities = np.array(serial.get('priorities', []), dtype=np.float64)
        obj.priorities = np.zeros(obj.capacity, dtype=np.float64)
        obj.priorities[:len(priorities)] = priorities

        # 恢复buffer内容
        obj.buffer = []
        for item in serial.get('buffer', []):
            s, a, r, ns, d, info = item
            s = np.array(s, dtype=np.float32)
            a = np.array(a, dtype=np.float32)
            ns = np.array(ns, dtype=np.float32)
            obj.buffer.append((s, a, r, ns, d, info))

        obj.target_counts = serial.get('target_counts', {})
        return obj

    def save(self, filename, compress=True):
        """保存buffer到文件"""
        try:
            serial = self.to_serializable()
            if compress:
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(serial, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filename, 'wb') as f:
                    pickle.dump(serial, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Replay buffer保存成功: {filename}")
        except Exception as e:
            logger.error(f"保存replay buffer失败: {e}")

    @staticmethod
    def load(filename, compress=True):
        """从文件加载buffer"""
        try:
            if compress:
                with gzip.open(filename, 'rb') as f:
                    serial = pickle.load(f)
            else:
                with open(filename, 'rb') as f:
                    serial = pickle.load(f)

            buf = PrioritizedReplayBufferOptimized.from_serializable(serial)
            logger.info(f"Replay buffer加载成功: {filename}, size={buf.size}")
            return buf
        except Exception as e:
            logger.error(f"加载replay buffer失败: {e}")
            return None


# ------------------- Actor/Critic 网络定义 -------------------
class UniversalActor(nn.Module):
    """通用Actor网络"""

    def __init__(self, state_dim=43, action_dim=3):  # 从63减少到43（30+3+10）
        super().__init__()

        # 特征提取层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
        )

        # 动作输出层
        self.action_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),

            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """前向传播"""
        features = self.feature_layers(state)
        action = self.action_layers(features)
        return action


class UniversalCritic(nn.Module):
    """通用Critic网络"""

    def __init__(self, state_dim=43, action_dim=3):
        super().__init__()

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
        )

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
        )

        # Q值计算层
        self.q_layers = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),

            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        """前向传播"""
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.q_layers(combined)
        return q_value


# ------------------- 模型保存/加载工具 -------------------
def save_model_atomic(save_path, actor, critic, optimizer_actor=None, optimizer_critic=None, extra=None):
    """修复版的模型保存函数"""

    # 🔧 在同一目录下创建临时文件
    save_dir = os.path.dirname(os.path.abspath(save_path)) or '.'
    os.makedirs(save_dir, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix='tmp_model_',
        dir=save_dir  # 关键修复：同一目录
    )
    os.close(tmp_fd)

    try:
        payload = {
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict()
        }

        if optimizer_actor is not None:
            payload['optimizer_actor_state_dict'] = optimizer_actor.state_dict()
        if optimizer_critic is not None:
            payload['optimizer_critic_state_dict'] = optimizer_critic.state_dict()
        if extra is not None:
            payload['extra'] = extra

        torch.save(payload, tmp_path)

        # 🔧 Windows兼容的文件替换
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(abs_save_path):
            os.remove(abs_save_path)
        shutil.move(tmp_path, abs_save_path)

        logger.info(f"模型保存成功: {save_path}")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


def load_model_if_exists(path, actor, critic, optimizer_actor=None, optimizer_critic=None, map_location='cpu'):
    """
    如果模型文件存在则加载，否则返回None
    """
    if not os.path.exists(path):
        logger.info(f"模型文件不存在: {path}")
        return None

    try:
        data = torch.load(path, map_location=map_location)

        if 'actor_state_dict' in data:
            actor.load_state_dict(data['actor_state_dict'])
        if 'critic_state_dict' in data:
            critic.load_state_dict(data['critic_state_dict'])
        if optimizer_actor is not None and 'optimizer_actor_state_dict' in data:
            optimizer_actor.load_state_dict(data['optimizer_actor_state_dict'])
        if optimizer_critic is not None and 'optimizer_critic_state_dict' in data:
            optimizer_critic.load_state_dict(data['optimizer_critic_state_dict'])

        logger.info(f"模型加载成功: {path}")
        return data.get('extra', None)

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None


# ------------------- 软更新函数 -------------------
def soft_update(target_net, source_net, tau=0.005):
    """
    软更新目标网络参数
    target = tau * source + (1 - tau) * target
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


# ------------------- 优化的训练函数 -------------------
def train_universal_ddpg_optimized(env, actor, critic, target_data, config):
    """
    优化版的DDPG训练函数
    - 内存使用优化
    - 改进的多进程支持
    - 增强的错误处理
    - 系统资源监控
    """

    # 解包配置参数
    num_episodes = config.get('num_episodes', 20000)
    max_steps = config.get('max_steps', 25)
    batch_size = config.get('batch_size', 64)
    gamma = config.get('gamma', 0.98)
    actor_lr = config.get('actor_lr', 1e-4)
    critic_lr = config.get('critic_lr', 1e-3)
    tau = config.get('tau', 0.005)
    buffer_size = config.get('buffer_size', 50000)
    n_parallel = config.get('n_parallel', 2)
    num_workers = config.get('num_workers', 4)
    use_multiprocessing = config.get('use_multiprocessing', True)
    save_every = config.get('save_every', 100)
    buffer_save_path = config.get('buffer_save_path', 'replay_buffer.pkl.gz')
    model_path = config.get('model_path', 'steel01_model.pth')

    logger.info("开始训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # 初始化优化器
    actor_optimizer = optim.AdamW(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=critic_lr)

    # 创建目标网络
    target_actor = UniversalActor(state_dim=43, action_dim=3)
    target_critic = UniversalCritic(state_dim=43, action_dim=3)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    target_actor.eval()
    target_critic.eval()

    # 加载已有模型和buffer
    extra = load_model_if_exists(model_path, actor, critic, actor_optimizer, critic_optimizer)
    start_episode = 0
    if extra and 'episode' in extra:
        start_episode = extra['episode']
        logger.info(f"从episode {start_episode}继续训练")

    # 初始化replay buffer
    buffer = None
    if os.path.exists(buffer_save_path):
        buffer = PrioritizedReplayBufferOptimized.load(buffer_save_path, compress=True)

    if buffer is None:
        buffer = PrioritizedReplayBufferOptimized(capacity=buffer_size)
        logger.info(f"创建新的replay buffer，容量: {buffer_size}")

    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_errors': [],
        'success_rate': deque(maxlen=100),
        'actor_losses': [],
        'critic_losses': [],
        'noise_levels': [],
        'opensees_failure_rate': [],
        'memory_usage': []
    }

    logger.info("开始训练...")

    # 多进程池（如果启用）
    pool = None
    if use_multiprocessing and num_workers > 1:
        try:
            pool = ProcessPoolExecutor(max_workers=num_workers)
            logger.info(f"启用多进程池，worker数量: {num_workers}")
        except Exception as e:
            logger.warning(f"创建进程池失败，将使用单进程: {e}")
            use_multiprocessing = False

    try:
        # 主训练循环
        for episode in range(start_episode, num_episodes):
            # 重置环境
            state = env.reset()
            episode_reward = 0.0
            episode_errors = []

            # 噪声水平（随训练逐渐减少）
            noise_level = 0.2 * (0.99 ** (episode // 100))

            # episode内的步骤循环
            for step in range(max_steps):
                # 使用actor网络选择主要动作
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0)
                    action_main = actor(s_t).cpu().numpy()[0]

                # 生成多个候选动作（包括主动作）
                actions = [action_main]
                for _ in range(n_parallel - 1):
                    noise = np.random.normal(0, noise_level, size=action_main.shape)
                    a = np.clip(action_main + noise, -1, 1)
                    actions.append(a)
                actions = np.array(actions)

                # 计算每个动作对应的新参数
                base_params = env.current_params.copy()
                params_list = []
                for a in actions:
                    new_p = []
                    for i, (p, ai) in enumerate(zip(base_params, a)):
                        delta = env.scale_factors[i] * ai
                        new_val = float(np.clip(p + delta, env.param_ranges[i][0], env.param_ranges[i][1]))
                        new_p.append(new_val)
                    params_list.append(new_p)

                # 并行计算滞回曲线
                if use_multiprocessing and pool is not None and num_workers > 1:
                    # 多进程版本
                    try:
                        # 将参数列表分块
                        chunks = []
                        k = min(num_workers, len(params_list))
                        chunk_size = max(1, (len(params_list) + k - 1) // k)
                        for i in range(0, len(params_list), chunk_size):
                            chunks.append(params_list[i:i + chunk_size])

                        # 提交任务
                        protocol = env.current_target['protocol'] if env.current_target else np.zeros(100)
                        futures = []
                        for ch in chunks:
                            futures.append(pool.submit(worker_hysteretic_batch_improved, (ch, protocol)))

                        # 收集结果
                        results_chunks = []
                        for fut in as_completed(futures):
                            try:
                                res = fut.result(timeout=30)  # 30秒超时
                                results_chunks.append(res)
                            except Exception as e:
                                logger.warning(f"Worker任务失败: {e}")
                                res = np.zeros((len(chunks[0]), len(protocol)))
                                results_chunks.append(res)

                        # 合并结果
                        batch_forces = np.vstack(results_chunks)

                    except Exception as e:
                        logger.warning(f"多进程计算失败，回退到单进程: {e}")
                        protocol = env.current_target['protocol'] if env.current_target else np.zeros(100)
                        batch_forces = env.hysteretic_curve_batch(params_list, protocol=protocol)
                else:
                    # 单进程版本
                    protocol = env.current_target['protocol'] if env.current_target else np.zeros(100)
                    batch_forces = env.hysteretic_curve_batch(params_list, protocol=protocol)

                # 计算奖励和经验
                done_flag = False
                for idx_par, (new_params, fitted_curve, act) in enumerate(zip(params_list, batch_forces, actions)):
                    # 计算误差
                    target_curve = env.current_target['curve'] if env.current_target else np.zeros_like(fitted_curve)
                    min_len = min(len(fitted_curve), len(target_curve))

                    if min_len == 0:
                        error_norm = float('inf')
                        relative_error = 1.0
                    else:
                        fitted_cut = fitted_curve[:min_len]
                        target_cut = target_curve[:min_len]
                        error_norm = np.linalg.norm(fitted_cut - target_cut)
                        relative_error = error_norm / (np.linalg.norm(target_cut) + 1e-6)

                    # 计算奖励
                    base_reward = np.exp(10 / relative_error)
                    # 精度奖励
                    if relative_error < 0.01:
                        precision_bonus = 10.0
                    elif relative_error < 0.05:
                        precision_bonus = 5.0
                    elif relative_error < 0.1:
                        precision_bonus = 1.0
                    else:
                        precision_bonus = 0.0

                    reward = base_reward + precision_bonus - 0.01
                    # 检查边界碰撞
                    hit_boundary = any([
                        (new_params[i] == env.param_ranges[i][0]) or
                        (new_params[i] == env.param_ranges[i][1])
                        for i in range(len(new_params))
                    ])

                    done = False
                    if hit_boundary:
                        reward -= 1.0
                        done = True
                    if relative_error < 0.05:
                        reward += 50.0
                        done = True

                    # 计算下一状态
                    oldp = env.current_params
                    env.current_params = new_params
                    next_state = env._get_state()
                    env.current_params = oldp

                    # 准备经验数据
                    state_arr = np.array(state, dtype=np.float32)
                    act_arr = np.array(act, dtype=np.float32)
                    next_state_arr = np.array(next_state, dtype=np.float32)
                    exp_info = {
                        'error_norm': error_norm,
                        'relative_error': relative_error,
                        'current_params': new_params.copy(),
                        'target_idx': env.current_target_idx,
                        'hit_boundary': hit_boundary
                    }

                    experience = (state_arr, act_arr, float(reward), next_state_arr, bool(done), exp_info)

                    # 添加到buffer
                    buffer.add(experience, error=relative_error)

                    # 主路径（第一个候选）更新环境状态
                    if idx_par == 0:
                        env.current_params = new_params
                        state = next_state
                        episode_reward += reward
                        if 'relative_error' in exp_info:
                            episode_errors.append(exp_info['relative_error'])
                        if done:
                            done_flag = True
                            break

                # 如果主路径完成，跳出步骤循环
                if done_flag:
                    break

                # 进行学习更新（当buffer有足够数据时）
                if buffer.size > max(batch_size, 256):
                    for _ in range(3):  # 减少到3次更新
                        batch = buffer.sample(batch_size)
                        if batch is None:
                            continue

                        states_b, actions_b, rewards_b, next_states_b, dones_b, indices_b, weights_b = batch

                        # Critic更新
                        with torch.no_grad():
                            next_actions = target_actor(next_states_b)
                            next_q_values = target_critic(next_states_b, next_actions)
                            target_q = rewards_b + gamma * next_q_values * (1 - dones_b)

                        current_q = critic(states_b, actions_b)
                        td_errors = (target_q - current_q).detach().cpu().numpy().flatten()

                        critic_loss = (weights_b * (current_q - target_q) ** 2).mean()
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
                        critic_optimizer.step()

                        # Actor更新
                        actor_actions = actor(states_b)
                        actor_loss = -(critic(states_b, actor_actions) * weights_b).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                        actor_optimizer.step()

                        # 软更新目标网络
                        soft_update(target_actor, actor, tau)
                        soft_update(target_critic, critic, tau)

                        # 更新buffer优先级
                        buffer.update_priorities(indices_b, td_errors)

                        # 记录损失
                        training_stats['actor_losses'].append(actor_loss.item())
                        training_stats['critic_losses'].append(critic_loss.item())

            # episode结束，记录统计信息
            training_stats['episode_rewards'].append(episode_reward)
            training_stats['episode_errors'].append(np.mean(episode_errors) if episode_errors else np.nan)
            training_stats['success_rate'].append(1 if (episode_errors and episode_errors[-1] < 0.01) else 0)
            training_stats['noise_levels'].append(noise_level)

            # OpenSees失败率
            op_total = env.opensees_total_count
            op_fail = env.opensees_failure_count
            failure_rate = op_fail / op_total if op_total > 0 else 0.0
            training_stats['opensees_failure_rate'].append(failure_rate)

            # 系统资源监控
            if episode % 100 == 0:
                resources = monitor_system_resources()
                if resources:
                    training_stats['memory_usage'].append(resources['memory_percent'])

                    # 内存过高时进行垃圾回收
                    if resources['memory_percent'] > 80:
                        logger.warning("内存使用率过高，执行垃圾回收")
                        force_garbage_collection()

            # 定期日志输出
            if episode % 10 == 0:
                pr = buffer.priorities[:buffer.size] if buffer.size > 0 else np.array([0.])
                logger.info(f"Ep {episode}/{num_episodes} | "
                            f"reward={episode_reward:.4f} | "
                            f"buffer_size={buffer.size} | "
                            f"priorities min/max/mean={pr.min():.3e}/{pr.max():.3e}/{pr.mean():.3e} | "
                            f"opensees_fail_rate={failure_rate:.3%}")

            # 定期保存
            if episode % save_every == 0 and episode > 0:
                extra = {
                    'episode': episode,
                    'stats_snapshot': {
                        k: (v[-1] if isinstance(v, list) and v else None)
                        for k, v in training_stats.items()
                    }
                }
                try:
                    save_model_atomic(model_path, actor, critic, actor_optimizer, critic_optimizer, extra=extra)
                except Exception as save_error:
                    logger.warning(f"保存失败，使用备用方法: {save_error}")
                    # 备用保存方法：直接保存，不使用临时文件
                    torch.save({
                        'actor_state_dict': actor.state_dict(),
                        'critic_state_dict': critic.state_dict(),
                        'optimizer_actor_state_dict': actor_optimizer.state_dict(),
                        'optimizer_critic_state_dict': critic_optimizer.state_dict(),
                        'extra': extra
                    }, model_path)
                    logger.info(f"备用保存成功: {model_path}")
                try:
                    buffer.save(buffer_save_path, compress=True)
                except Exception as e:
                    logger.error(f"保存buffer失败: {e}")

            # 详细诊断（每100个episode）
            if episode % 100 == 0 and episode > 0:
                if training_stats['critic_losses']:
                    recent_critic_loss = np.mean(training_stats['critic_losses'][-50:])
                    logger.info(f"最近critic_loss均值: {recent_critic_loss:.6f}")
                logger.info(f"OpenSees 统计 - 总计/失败: {op_total}/{op_fail} (失败率={failure_rate:.3%})")

                # 成功率统计
                if training_stats['success_rate']:
                    recent_success_rate = np.mean(list(training_stats['success_rate']))
                    logger.info(f"最近100个episode成功率: {recent_success_rate:.3%}")

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")
    finally:
        # 清理资源
        if pool is not None:
            logger.info("关闭进程池...")
            pool.shutdown(wait=True)

        # 最终保存
        logger.info("保存最终模型和buffer...")
        extra = {'episode': episode if 'episode' in locals() else num_episodes, 'finished': True}
        try:
            save_model_atomic(model_path, actor, critic, actor_optimizer, critic_optimizer, extra=extra)
        except Exception as save_error:
            logger.warning(f"保存失败，使用备用方法: {save_error}")
            # 备用保存方法：直接保存，不使用临时文件
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': critic_optimizer.state_dict(),
                'extra': extra
            }, model_path)
            logger.info(f"备用保存成功: {model_path}")
        try:
            buffer.save(buffer_save_path, compress=True)
        except Exception as e:
            logger.error(f"保存最终buffer失败: {e}")

    return training_stats


# ------------------- 参数识别函数 -------------------
def identify_curve_parameters(actor, target_curve, protocol,
                              param_ranges=None, scale_factors=None,
                              max_iterations=30, tolerance=0.01, verbose=True):
    """
    使用训练好的actor网络识别曲线参数

    参数:
        actor: 训练好的actor网络
        target_curve: 目标力曲线
        protocol: 位移历程
        param_ranges: 参数范围
        scale_factors: 参数调整比例
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        verbose: 是否显示详细信息

    返回:
        list: 识别出的参数 [fy, E, b]
    """
    if param_ranges is None:
        param_ranges = [[1, 1000], [1, 50000], [0.01, 0.3]]
    if scale_factors is None:
        scale_factors = [10, 500, 0.005]

    # 创建临时环境
    temp_env = MultiTargetHysteresisEnvOptimized([])
    temp_env.param_ranges = param_ranges
    temp_env.scale_factors = scale_factors
    temp_env.current_target = {
        'curve': target_curve,
        'protocol': protocol,
        'params': [0, 0, 0]
    }

    # 初始参数（使用范围中点）
    current_params = [(r[0] + r[1]) / 2 for r in param_ranges]

    actor.eval()

    logger.info("开始参数识别...")

    for iteration in range(max_iterations):
        # 设置当前参数并获取状态
        temp_env.current_params = current_params
        state = temp_env._get_state()

        # 使用actor网络预测动作
        with torch.no_grad():
            action = actor(torch.FloatTensor(state).unsqueeze(0)).cpu().numpy()[0]

        # 更新参数
        new_params = []
        for i, (p, a) in enumerate(zip(current_params, action)):
            new_p = p + scale_factors[i] * a
            new_p = np.clip(new_p, param_ranges[i][0], param_ranges[i][1])
            new_params.append(new_p)

        # 计算拟合曲线和误差
        fitted_curve = temp_env.hysteretic_curve_single(new_params, protocol)
        error = np.linalg.norm(fitted_curve - target_curve)
        relative_error = error / (np.linalg.norm(target_curve) + 1e-6)

        if verbose:
            logger.info(f"迭代 {iteration + 1}: "
                        f"params=[{new_params[0]:.2f}, {new_params[1]:.2f}, {new_params[2]:.4f}] "
                        f"相对误差={relative_error:.6f}")

        # 检查收敛
        if relative_error < tolerance:
            logger.info(f"参数识别收敛! 相对误差={relative_error:.6f}")
            break

        current_params = new_params

    return current_params


# ------------------- 训练进度可视化 -------------------
def plot_training_progress(stats, save_path='training_progress.png'):
    """
    绘制训练进度图表

    参数:
        stats: 训练统计数据
        save_path: 保存路径
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 奖励曲线
    ax = axes[0, 0]
    rewards = stats['episode_rewards']
    if rewards:
        ax.plot(rewards, alpha=0.3, label='Raw Reward', color='blue')
        if len(rewards) > 100:
            window = min(100, len(rewards) // 10)
            ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), ma, label=f'{window}-Episode MA', color='red', linewidth=2)
    ax.set_title('Episode Rewards')  # 英文标题
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True)

    # 误差曲线
    ax = axes[0, 1]
    if stats['episode_errors']:
        errs = np.array([e for e in stats['episode_errors'] if not np.isnan(e)], dtype=np.float64)
        if len(errs) > 0:
            ax.plot(errs, alpha=0.6, color='orange')
            ax.set_title('Episode Mean Relative Error')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Relative Error')
            ax.set_yscale('log')
            ax.grid(True)

    # 成功率
    ax = axes[0, 2]
    if stats['success_rate']:
        sr = list(stats['success_rate'])
        x = range(len(sr))
        ax.plot(x, sr, alpha=0.7, color='green')
        ax.set_title('Success Rate (Sliding Window)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.05)
        ax.grid(True)

    # 损失曲线
    ax = axes[1, 0]
    if stats['actor_losses'] and stats['critic_losses']:
        actor_losses = stats['actor_losses']
        critic_losses = stats['critic_losses']
        ax.plot(actor_losses, alpha=0.6, label='Actor Loss', color='red')
        ax.plot(critic_losses, alpha=0.6, label='Critic Loss', color='blue')
        ax.set_title('Network Losses')
        ax.set_xlabel('Update Steps')
        ax.set_ylabel('Loss Value')
        ax.legend()
        ax.grid(True)

    # OpenSees失败率
    ax = axes[1, 1]
    if stats.get('opensees_failure_rate'):
        fail_rates = stats['opensees_failure_rate']
        ax.plot(fail_rates, alpha=0.7, color='purple')
        ax.set_title('OpenSees Failure Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Failure Rate')
        ax.grid(True)

    # 内存使用情况
    ax = axes[1, 2]
    if stats.get('memory_usage'):
        memory = stats['memory_usage']
        ax.plot(memory, alpha=0.7, color='brown')
        ax.set_title('Memory Usage')
        ax.set_xlabel('Monitoring Point')
        ax.set_ylabel('Memory Usage (%)')
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Warning Line (80%)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Training progress saved to: {save_path}")


# ------------------- 获取优化配置 -------------------
# 在main函数中，将配置改为：
def get_emergency_fast_config():
    return {
        'num_episodes': 1000,         # 先测试1000回合
        'max_steps': 10,              # 减少步数
        'batch_size': 64,
        'actor_lr': 1e-3,             # 🔧 增大10倍
        'critic_lr': 5e-3,            # 🔧 增大5倍
        'tau': 0.02,                  # 🔧 增大4倍
        'buffer_size': 5000,          # 🔧 减小10倍
        'n_parallel': 1,              # 🔧 单候选
        'num_workers': 1,             # 🔧 单进程
        'use_multiprocessing': False, # 🔧 禁用多进程
        'save_every': 50,
        'model_path': 'fast_test_model.pth'
    }


# ------------------- 主函数 -------------------
def main():
    """主函数"""
    # 数据文件路径（需要根据实际情况修改）
    txt_path = r"D:\Charles\PycharmProjects\PythonProject\.venv\vacation_task\final\data_files\new_data\10steel01data.txt"
    excel_path = r"D:\Charles\PycharmProjects\PythonProject\.venv\vacation_task\final\data_files\(250,15000,0.1)3圈steel01.xlsx"

    logger.info("=" * 60)
    logger.info("Steel01 参数识别系统 (优化版 - 8核16GB)")
    logger.info("=" * 60)

    # 系统信息
    if PSUTIL_AVAILABLE:
        logger.info("系统配置信息:")
        logger.info(f"  CPU核心数: {psutil.cpu_count()}")
        logger.info(f"  总内存: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
        logger.info(f"  可用内存: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")

    # 模式选择
    print("\n请选择运行模式:")
    print("[1] 训练新模型")
    print("[2] 使用已有模型进行参数识别")
    print("[3] 继续训练已有模型")

    mode = input("请输入选择 (1/2/3): ").strip()

    if mode == '1' or mode == '3':
        # 训练模式
        logger.info(f"{'训练新模型' if mode == '1' else '继续训练已有模型'}")

        # 检查数据文件
        if not os.path.exists(txt_path):
            logger.error(f"训练数据文件不存在: {txt_path}")
            logger.error("请检查文件路径是否正确")
            return

        # 加载训练数据
        target_data = load_training_data_from_txt(txt_path)
        if not target_data:
            logger.error("无法加载训练数据")
            return

        # 创建环境和网络
        env = MultiTargetHysteresisEnvOptimized(target_data, feature_dim=30)
        actor = UniversalActor(state_dim=43, action_dim=3)
        critic = UniversalCritic(state_dim=43, action_dim=3)

        # 获取优化配置
        config = get_emergency_fast_config()
        # config = get_test_config()

        # 用户确认配置
        print("\n训练配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        confirm = input("\n是否使用此配置开始训练? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("训练取消")
            return

        # 开始训练
        try:
            training_stats = train_universal_ddpg_optimized(env, actor, critic, target_data, config)

            # 绘制训练进度
            plot_training_progress(training_stats)
            logger.info("训练完成!")

        except Exception as e:
            logger.exception(f"训练过程中发生错误: {e}")

    elif mode == '2':
        # 参数识别模式
        logger.info("参数识别模式")

        # 查找可用的模型文件
        model_files = glob.glob('steel01_model*.pth')
        # model_files = glob.glob('test_model.pth')
        if not model_files:
            logger.error("未找到训练好的模型文件")
            logger.error("请先使用模式1训练模型")
            return

        # 选择模型文件
        if len(model_files) == 1:
            model_path = model_files[0]
            logger.info(f"使用模型文件: {model_path}")
        else:
            logger.info("发现多个模型文件:")
            for i, f in enumerate(model_files):
                logger.info(f"  {i + 1}. {f}")
            try:
                choice = int(input("请选择模型文件编号: ").strip()) - 1
                model_path = model_files[choice]
            except (ValueError, IndexError):
                logger.error("无效选择")
                return

        # 加载模型
        actor = UniversalActor(state_dim=43, action_dim=3)
        critic = UniversalCritic(state_dim=43, action_dim=3)
        extra = load_model_if_exists(model_path, actor, critic)

        if extra is None:
            logger.error("模型加载失败")
            return

        # 检查测试数据文件
        if not os.path.exists(excel_path):
            logger.error(f"测试数据文件不存在: {excel_path}")
            logger.error("请检查文件路径是否正确")
            return

        # 加载测试数据
        logger.info(f"加载测试数据: {excel_path}")
        displacement, force = load_test_data_from_excel(excel_path)

        if len(displacement) == 0 or len(force) == 0:
            logger.error("测试数据加载失败")
            return

        logger.info(f"测试数据点数: {len(displacement)}")

        # 参数识别
        logger.info("开始参数识别...")
        try:
            params = identify_curve_parameters(
                actor, force, displacement,
                max_iterations=50,
                tolerance=0.01,
                verbose=True
            )

            logger.info("=" * 40)
            logger.info("参数识别结果:")
            logger.info(f"  屈服强度 (fy): {params[0]:.2f}")
            logger.info(f"  弹性模量 (E):  {params[1]:.2f}")
            logger.info(f"  应变硬化比 (b): {params[2]:.4f}")
            logger.info("=" * 40)

            # 验证结果
            env = MultiTargetHysteresisEnvOptimized([])
            env.current_target = {'protocol': displacement, 'curve': force}
            env.current_params = params
            predicted_force = env.hysteretic_curve_single(params, displacement)

            # 计算误差
            error = np.linalg.norm(predicted_force - force)
            relative_error = error / (np.linalg.norm(force) + 1e-6)
            logger.info(f"验证误差: 相对误差 = {relative_error:.6f}")

            # 绘制对比图
            plt.figure(figsize=(12, 8))
            plt.plot(displacement, force, 'b-', linewidth=2, label='target_curve', alpha=0.8)
            plt.plot(displacement, predicted_force, 'r--', linewidth=2, label='fitting_curve', alpha=0.8)
            plt.xlabel('displacement')
            plt.ylabel('force')
            plt.title(f'Steel01 comparison of parameter recognition results\n'
                      f'fy={params[0]:.2f}, E={params[1]:.2f}, b={params[2]:.4f}\n'
                      f'Relative error={relative_error:.6f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.exception(f"参数识别过程中发生错误: {e}")

    else:
        logger.error("无效的模式选择")


# ------------------- 程序入口 -------------------
if __name__ == "__main__":
    # Windows多进程支持
    import multiprocessing

    multiprocessing.freeze_support()

    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.exception(f"程序执行中发生未处理的错误: {e}")
    finally:
        logger.info("程序结束")