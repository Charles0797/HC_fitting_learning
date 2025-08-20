import numpy as np
import openseespy.opensees as ops


def make_protocol(fy, E, pts_per_cycle=2000):
    uy = fy / E
    base = np.array([0, 1, 0, -1, 0], dtype=float)
    t = np.linspace(0, 1, base.size)
    t_fine = np.linspace(0, 1, pts_per_cycle)

    cycles = []
    for alpha in [1, 2, 3]:  # 固定三圈：1,2,3倍
        umax = alpha * uy
        cycle = np.interp(t_fine, t, base * umax)
        cycles.append(cycle)
    return np.concatenate(cycles)


def cyclic(protocol, params):
    fy, E, b = params
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0);
    ops.node(2, 0.0)
    ops.fix(1, 1)
    ops.uniaxialMaterial("Steel01", 1, fy, E, b)
    ops.element("twoNodeLink", 1, 1, 2, "-mat", 1, "-dir", 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0)
    ops.algorithm("Newton")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandSPD")
    ops.integrator("DisplacementControl", 2, 1, 0.0)
    ops.analysis("Static")

    forces = np.zeros_like(protocol)
    for i, u_tot in enumerate(protocol):
        du = u_tot if i == 0 else u_tot - protocol[i - 1]
        ops.integrator("DisplacementControl", 2, 1, du)
        if ops.analyze(1) == 0:
            forces[i] = ops.eleForce(1)[1]
        else:
            forces[i] = np.nan
    ops.wipe()
    return forces


if __name__ == "__main__":
    N = 10 # 样本数量
    pts_per_cycle = 2000  # 每圈点数
    n_cycles = 3  # 固定3圈
    data_points = pts_per_cycle * n_cycles  # 位移/力数据点数
    total_rows = 10 + 2 * data_points  # 总行数 = 10(参数+空行) + 6000位移 + 6000力

    # 生成参数列表
    params_list = []
    for i in range(N):
        fy = np.random.uniform(1, 1000)
        E = np.random.uniform(1, 50000)
        b = np.random.uniform(0.01, 0.3)
        params_list.append((fy, E, b))

    # 初始化数据矩阵（总行数 x 样本数）
    data_matrix = np.empty((total_rows, N))
    data_matrix[:] = np.nan  # 初始化为NaN

    # 生成所有样本数据
    for col_idx, (fy, E, b) in enumerate(params_list):
        # 生成位移协议
        prot = make_protocol(fy, E, pts_per_cycle)
        # 计算力响应
        forces = cyclic(prot, (fy, E, b))

        # 填充数据矩阵
        data_matrix[0, col_idx] = fy  # 第1行: fy
        data_matrix[1, col_idx] = E  # 第2行: E
        data_matrix[2, col_idx] = b  # 第3行: b
        # 第4-10行保持NaN（空行）
        # 第11-6010行: 位移数据
        data_matrix[10:10 + data_points, col_idx] = prot
        # 第6011-12010行: 力数据
        data_matrix[10 + data_points:10 + 2 * data_points, col_idx] = forces

    # 写入文件
    with open("10steel01data.txt", "w") as f:
        # 逐行写入数据
        for row in data_matrix:
            # 将每行数据转换为字符串，NaN替换为空字符串
            line = []
            for val in row:
                if np.isnan(val):
                    line.append("")
                else:
                    line.append(f"{val:.6f}")
            f.write("\t".join(line) + "\n")

    print(f"已生成并保存 10steel01data.txt")
    print(f"格式: 每列一个样本，共{N}列")
    print(f"行 1: fy, 行 2: E, 行 3: b")
    print(f"行 4-10: 空")
    print(f"行 11-{10 + data_points}: 位移数据 ({data_points}点)")
    print(f"行 {11 + data_points}-{10 + 2 * data_points}: 力数据 ({data_points}点)")
    print(f"总行数: {total_rows}")