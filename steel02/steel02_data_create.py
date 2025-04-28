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
    # 修改材料定义部分
    ops.uniaxialMaterial("Steel02", 1, *params[:10])
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
    original_protocol = np.array([0.00, 0.01, 0.02, 0.03, 0.02, 0.01, 0.00, -0.01, -0.02, -0.03, -0.02, -0.01, 0.00])
    # 使用插值生成2000个数据点
    base = np.interp(
        np.linspace(0, 1, 2000),
        np.linspace(0, 1, len(original_protocol)),
        original_protocol
    )
    protocol = []
    for amp in [3, 5, 7]:  # 三圈不同幅值
        protocol += [x * amp / 3 for x in base]

    # Steel02材料参数 (需包含更多参数)
    # 正确参数设置示例 (参数值需在合理区间)
    params = [
        200.0,  # fy (屈服强度, MPa)
        200000,  # E (弹性模量, MPa 需确保与钢材匹配)
        0.01,  # b (应变硬化比，建议0.005~0.03)
        15.0,  # R0 (各向同性硬化参数，推荐15-20)
        0.925,  # cR1 (包辛格效应参数1)
        0.15,  # cR2 (包辛格效应参数2)
        0.0,  # a1 (滞回参数)
        1.0,  # a2 (滞回参数)
        0.0,  # a3 (滞回参数)
        1.0  # a4 (滞回参数)
    ]

    # 调用函数进行计算
    forces_single = cyclic(protocol, params)

    # df = pd.DataFrame({'displacement': protocol, 'force': forces_single})
    # df.to_excel('3圈steel02.xlsx', index=False)

    # 材料的荷载-位移曲线
    plt.figure(figsize=(6, 4))

    plt.plot(protocol, forces_single, linestyle='-', color='b')
    plt.xlabel("displacement")
    plt.ylabel("force")
    plt.title("hy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()