import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def cyclic(protocol, params):
    # 建立一维单自由度模型
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 0.0)
    ops.fix(1, 1)

    # 定义单一钢材材料
    ops.uniaxialMaterial("Steel01", 1, params[0], params[1], params[2])
    # 建立两节点链接单元
    ops.element("twoNodeLink", 1, 1, 2, "-mat", 1, "-dir", 1)

    # 定义时程与荷载
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0)

    # 分析设置
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", 2, 1, 1)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandSPD")
    ops.analysis("Static")

    protocol_size = len(protocol)
    result = [0. for i in range(protocol_size)]

    for i in range(protocol_size):
        # 计算当前步位移增量
        disp = protocol[i] if i == 0 else protocol[i] - protocol[i - 1]
        ops.integrator("DisplacementControl", 2, 1, disp)
        ok = ops.analyze(1)
        if ok == 0:
            forces = ops.eleForce(1)
            result[i] = forces[1]
        else:
            result[i] = 0.0  # 若分析失败，记录0
    ops.wipe()
    return result
if __name__ == "__main__":
    # 生成三圈递增幅值的位移协议
    original_protocol = np.array([0., 1., 2., 3., 2., 1., 0., -1., -2., -3., -2., -1., 0.])
    base = np.interp(
        np.linspace(0, 1, 2000),
        np.linspace(0, 1, len(original_protocol)),
        original_protocol
    )
    protocol = []
    for amp in [3, 5, 7]:  # 三圈不同幅值
        protocol += [x * amp / 3 for x in base]

    #参数：[屈服强度, 弹性模量, 硬化比]
    params = [250., 150., 0.1]

    # 调用函数进行计算
    forces_single = cyclic(protocol, params)

    df = pd.DataFrame({'displacement': protocol, 'force': forces_single})
    df.to_excel('(250,150)3圈steel01.xlsx', index=False)

    # 材料的荷载-位移曲线
    plt.figure(figsize=(6, 4))

    plt.plot(protocol, forces_single, linestyle='-', color='b')
    plt.xlabel("displacement")
    plt.ylabel("force")
    plt.title("hy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
