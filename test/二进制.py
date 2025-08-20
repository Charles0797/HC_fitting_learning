import numpy as np
import torch
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

def save_binary(data, filename):
    """保存为PyTorch二进制格式"""
    torch.save(data, filename)
    print(f"已保存二进制文件: {filename}")


if __name__ == "__main__":
    N = 100
    pts_per_cycle = 2000
    n_cycles = 3
    data_points = pts_per_cycle * n_cycles

    # 创建数据集
    dataset = {
        "params": np.empty((N, 3)),
        "displacement": np.empty((N, data_points)),
        "force": np.empty((N, data_points))
    }

    # 生成样本
    for i in range(N):
        fy = np.random.uniform(1, 1000)
        E = np.random.uniform(1, 50000)
        b = np.random.uniform(0.01, 0.3)

        prot = make_protocol(fy, E, pts_per_cycle)
        forces = cyclic(prot, (fy, E, b))

        dataset["params"][i] = [fy, E, b]
        dataset["displacement"][i] = prot
        dataset["force"][i] = forces

    # 保存主数据集（二进制）
    torch.save(dataset, "steel01_dataset.pt")

    # 保存文本样本（前5个样本）
    with open("sample_data.txt", "w") as f:
        f.write("Sample\tfy\tE\tb\n")
        for i in range(min(5, N)):
            fy, E, b = dataset["params"][i]
            f.write(f"{i + 1}\t{fy:.6f}\t{E:.6f}\t{b:.6f}\n")
            f.write("Displacement:\n")
            f.write("\n".join(f"{x:.6f}" for x in dataset["displacement"][i]))
            f.write("\nForce:\n")
            f.write("\n".join(f"{x:.6f}" for x in dataset["force"][i]))
            f.write("\n\n")

    print(f"已生成数据集: {N}个样本")
    print(f"主文件: steel01_dataset.pt (PyTorch二进制格式)")
    print(f"示例文件: sample_data.txt (文本格式)")