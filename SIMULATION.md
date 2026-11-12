# Pi Plus Pro 仿真导航与基础运控

> 2026-11-11：当前仿真与机载入口已共用 `navigation/`，AMP 固定为真机 Pro 版本。
> 当前代码、测试结果和未通过项以 [共享导航状态](SHARED_NAVIGATION_STATUS.md) 为准；下文旧策略成绩保留作历史记录。


本文记录原有真值导航基线。新增的相机驱动版本使用独立入口 `scripts/run_visual.py`，
传感器边界、运行方式与验证记录见 [VISUAL_NAVIGATION.md](VISUAL_NAVIGATION.md)。

## rt1 机载导航复跑

在 macOS 上可以打开实时 MuJoCo 窗口。下例保留 01 起点和三箱布置，初始机身偏航为 0.14 rad（约 8°）。主视角跟随机器人，左下角显示头部 RGB 画面。去掉 `--spawn` 会使用 01 的标准正向起步姿态。运行结束后窗口保留最终姿态，可直接关闭。

```sh
# 现场照片中的折角入弯（默认场地）
OPENBLAS_NUM_THREADS=1 .venv-shared/bin/mjpython scripts/run_visual.py --viewer --start 01 --spawn .55 -1.25 .14 --bend-profile corner
# 原来的平滑入弯
OPENBLAS_NUM_THREADS=1 .venv-shared/bin/mjpython scripts/run_visual.py --viewer --start 01 --spawn .55 -1.25 .14 --bend-profile smooth
```

选中的场地同时决定头部相机看到的白线和独立裁判使用的跑道边界。旧记录缺少 `bend_profile` 字段时，回放和评估使用原来的平滑版。[平滑入弯近景](output/athletics/bend_entry.png) · [折角入弯近景](output/athletics_corner/bend_entry.png)

01 起点、初始偏航 0.14 rad 的两次 GUI 试跑使用相同的三箱布局、rt1 导航、相机时序和噪声种子。两次均跑到 300 s，均未到达箱区。

| 入弯 | 沿赛道前进 | 停步时的导航记录 |
|---|---:|---|
| [折角](output/sim/rt1_gui_corner_01_20261111_154350/summary.json) | 2.19 m | 11.39 s 后始终没有识别出有效车道，最终为 `lost_vision`。基座停在外侧折点前约 0.26 m，头部画面可见一条带折角的白线。[末帧](output/sim/rt1_gui_corner_01_20261111_154350/camera_last_rgb.png) |
| [平滑](output/sim/rt1_gui_01_20261111_152856/summary.json) | 3.51 m | 机器人进入大弯，末帧仍有车道拟合，前方地面深度覆盖检查返回 `unknown_depth`。 |

`scripts/run_visual.py` 默认使用 2026-11-11 从 `shared_nav_20261111_rt1` 核对的 `OnboardNavigation`。普通赛道由当前白线控制，地面跟踪在后台更新，进入箱区或终点时转入同步跟踪。三份与仓内旧版不同的源码保存在 `vendor/pi_plus_pro/rt1/`，只调整了两处包导入。运行清单会核对这三份源码和其余共享依赖的机载哈希。旧基线可通过 `--navigation-profile shared` 运行。

D435i 按真机 640×480 内参渲染，使用相同的 OpenCV 映射得到 256×192 RGB-D，深度量化为 1 mm。相机配置为 6 Hz。视觉时序取自 `20261111_070539Z_88bb2a6c` 的 1,106 个已处理帧，保留实际取帧间隔、接收时间、开始处理时间、重映射和感知耗时，以及后台结果可读的时机。仿真只渲染机载循环实际消费的帧。姿态与画面在下一个 20 ms 物理控制周期采样。图像使用 1 s 时效上限，绕箱控制可指定更早的制动时刻。发布器和监督器按同相位 50 Hz 模拟，监督器使用 AMP 策略反馈做最终速度适配。请求链路假设健康且无人接管。

2 s 零速站立后开始播放机载时序。超过约 236 s 时重复该段采样间隔。机载样本只覆盖普通巡线，箱区的额外计算延迟尚无同版样本。反馈静差、电机和足底接触仍按现有 Pro 物理模型运行。

```sh
r=output/sim/rt1_01_three_boxes_20261111
OPENBLAS_NUM_THREADS=1 .venv-shared/bin/python scripts/run_visual.py --bend-profile smooth --start 01 --duration 300 --seed 1 --record-sensors 12 --video --output "$r"
.venv-shared/bin/python scripts/evaluate_visual.py "$r"
.venv-shared/bin/python scripts/replay_navigation.py "$r/sensor_replay.npz" --map-observations "$r/observations.json"
```

本次三箱从 01 正向起步，目标 0.22 m/s，噪声种子 1。300 s 后净赛道进度为 0 m，最大正向进度约 1 cm。1,384 个已处理画面均由当前白线控制，其中 1,376 帧为 ready、6 帧图像过期、2 帧站立恢复。第 40–60 秒前向速度请求、监督器输出和策略滤波后输入均为 0.22，基座净位移约 0.014 mm。按保存的 50 Hz 物理状态重算，1,000 个采样点的双脚均接触地毯，左、右踝高度变化范围分别为 0.0065 和 0.0036 mm。这轮没有形成交替迈步，也未进入绕箱控制。[起步与足端统计](output/sim/rt1_01_three_boxes_20261111/start_diagnosis.json) · [独立裁判](output/sim/rt1_01_three_boxes_20261111/evaluation.json) · [前 12 s 传感器回放](output/sim/rt1_01_three_boxes_20261111/replay_check.json)

## Pro 头部与相机

默认 MuJoCo 场景使用随真机提供的 `PiPlusPro_SF_12L8A0G2H0W_Z_261010.xml`、网格、刚体惯量、关节限位和碰撞形状。20 个身体关节仍由原厂 AMP 控制。两个头部关节按任务记录中的实测反馈初始化并通过独立的姿态约束保持。真实头部伺服的静差和模式切换瞬态尚未建模。Pro CAD 导出省略了电机折算惯量，仿真沿用上游 Mini 对三类驱动的估计值。接触参数和驱动响应仍需实机辨识。

D435i 使用实测焦距、主点及分辨率。导航相机跟随 `torso_link → head_yaw_link → head_pitch_link → camera_link`。光学朝向由 standby 的 18 帧地面法向、头部反馈和同时段 IMU 姿态估计。此处假定 IMU 与机身轴对齐，光心近似位于 CAD 镜头前表面。地面平面不能单独确定镜头平移或绕重力轴的方向。[安装参数](sim/pro_camera_mount.json)

01 起步时复现标定的机身横滚 −1.26°、俯仰 +5.02°，头部 pitch 0.63837 rad、yaw 0。渲染深度独立拟合得到 43.95°（实测 43.97°）。同一机器人进入零速 AMP 自由站立 2 s 后，仿真深度俯角为 38.99°。真机行走时的相机俯角仍待相应工况的测量。[站姿核对](output/sim/pro_calibrated/standing/standing.json)

```sh
export OPENBLAS_NUM_THREADS=1
.venv-shared/bin/python scripts/check_pro_standing.py --output output/sim/pro_calibrated/standing
.venv-shared/bin/python scripts/run_visual.py --bend-profile smooth --navigation-profile shared --start 01 --duration 180 --video --output output/sim/pro_calibrated/official
.venv-shared/bin/python scripts/evaluate_visual.py output/sim/pro_calibrated/official
```

以下数据来自头部模型同步时的回归。地面纹理窗口进一步适配待机与站立姿态后，同一场景的闭环进度和定位误差见 [Pro 任务头姿下的建图验证](SHARED_NAVIGATION_STATUS.md#pro-任务头姿下的建图验证)。

三组 01 起点试跑的共享导航源码哈希相同，均开启种子 1 的传感器噪声，速度目标为 0.22 m/s。偏航对照只改变初始机身朝向，头部 yaw 始终设为 0。进度由独立裁判按 21.56 m 的赛道中心线计算。

| 场景 | 结果 |
|---|---|
| 正向起步，三箱 | 63.14 s 因持续未走动而停车，净进度约 0.01 m。[裁判记录](output/sim/pro_calibrated/official/evaluation.json) |
| 初始机身偏航 +8.0°，三箱 | 64.78 s 左大腿触碰 02 号箱，进度 15.73 m（72.9%）。[裁判记录](output/sim/pro_calibrated/heading8/evaluation.json) |
| 同一偏航，无箱巡线 | 81.54 s 实际过线，81.62 s 视觉确认，随后观察 3 s，完整圈通过核验。[轨迹与终点评估](output/sim/pro_calibrated/lane_only/evaluation.json) |

三组运行的头部 pitch 均在 0.63835–0.63850 rad 内，yaw 绝对值小于 0.00009 rad。[逐组状态统计](output/sim/pro_calibrated/head_during_runs.json)

身体直行回归在 Pro 上复现了起步问题。零速站立 2 s 后，向 AMP 下发 0.4 m/s 直行指令 3 s，基座仅前进 0.028 m，测试要求超过 0.35 m。[定向测试记录](output/sim/pro_calibrated/tests.log)

## 2026-11-10：真机基础运控同步

原仿真与真机不一致，现已将 `sim.robot.Robot` 的默认策略切换到真机当前版本。
`scripts/run_sim.py`、`scripts/run_visual.py` 共用此入口。同步源为真机
`/home/nvidia/action_library/policy/amp/`，原始文件、导出的 CPU 权重及哈希保存在
`vendor/pi_plus_pro/`；审计证据在 `output/deployment/loco_sync_20261110/`。

| 项目 | 原仿真 | 当前默认仿真 / 真机配置 |
|---|---|---|
| 权重 | 上游 `model_40000.pt` | 当前 `amp.trt` 中的同一组 FP32 权重，策略标识 `pi_plus_amp_1110_policy_1.rknn` |
| 观测 / 历史 | 69 × 5，观测限幅 ±100 | 69 × 5，观测限幅 ±18，旧帧到新帧，初始历史为零 |
| 动作 | 限幅 ±100，缩放 0.25 | 限幅 ±10，缩放 0.25，各关节额外缩放为 1 |
| 速度入口 | 直接送入策略 | Soccer / normal，限幅后 0.3 低通，再执行死区 |
| 死区 | 无控制器死区 | 平移 0.08 m/s、转向 0.17 rad/s，死区处理结果参与下一帧滤波 |
| PD / 物理频率 | 500 Hz | 1000 Hz；策略频率仍为 50 Hz |
| PD 增益 | 上游较小阻尼 | 使用实机 AMP 的 `kp`、`kd`，按关节名映射 |

原厂运行的是 Jetson TensorRT 引擎，配置中的 `.rknn` 文件名不是实际推理后端。
设备上没有对应原始 ONNX/PT，故从已校验 SHA-256 的 TensorRT 文件提取原有 FP32 张量，
使用相同的 `345→512→256→128→20` 网络与 ELU 激活在 CPU 推理，没有训练或拟合新策略。
`scripts/export_device_amp.py` 只接受这一个已审计引擎的哈希，拒绝对其他版本套用字节布局。

验证结果：129 组独立 Jetson 推理输入的最大绝对误差为 `1.3352e-5`；另从正在运行的
原厂控制器只读订阅连续 50 帧调试数据，按序号核对历史堆叠、上一帧动作、20 关节目标映射，
本地策略输出最大绝对误差为 `2.3842e-7`。本次没有向真机发送速度、电机或模式切换指令。
站立、0.4 m/s 直行及双向带前进转弯各进行了 2 秒稳定 + 5 秒指令仿真，均未摔倒。
详细结果在 `output/sim/device_amp_sync_20261110/motion_checks.json`。

**旧导航参数需要重新标定。** 原默认 0.22–0.25 m/s 从零开始时，经 0.3 滤波得到
0.066–0.075 m/s，被 0.08 m/s 死区置零，并且不会逐帧积累。因此旧的默认导航目前不能起步；
原 0.12 m/s 侧移下限及较小转向指令也不能继续沿用。新增 `policy_cmd_vx/vy/wz` 遥测列记录
实际送入网络的速度，`summary.json` / 视觉 `run_manifest.json` 记录策略、配置哈希与频率。
完成基础运控同步不代表当前策略已重新通过整圈导航。

```sh
# 当前真机运控的短时真值导航检查；到达时限会报告 duration_limit
.venv/bin/python scripts/run_sim.py --speed .4 --duration 10 --output output/sim/pro_smoke
# 明确指定旧权重，复查历史基线（不是当前真机版本）
.venv/bin/python scripts/run_sim.py --policy vendor/mini_pi_plus_amp/Exported_policy/model_40000.pt
.venv/bin/python -m unittest scripts.test_device_amp scripts.test_sim scripts.test_visual scripts.test_camera_profile
```

同步范围是 AMP 身体运控，使用原厂默认 Soccer / normal 速度模式。手柄模式按钮、
真机启动状态机、看门狗、头部控制不属于此仿真适配器。旧 MJCF 与真机的关节零点通过
相对默认姿态的关节位移转换；没有把真机肩部编码器偏置直接填进旧模型的关节角。
此次 11 月 10 日的对照使用旧 Mini 几何和机身固定相机。当前 Pro 场景的假设和结果见上节。

## 同步前的历史完成范围

已在现有田径场地上运行真实 MuJoCo 关节动力学：导航输出身体坐标系 `vx, vy, wz`，
高擎 Mini Pi Plus AMP 输出 20 关节目标，PD 控制驱动电机。
三个起点均完成逆时针一圈、绕过三个箱体，并在终点后继续仿真 3 秒检查减速停车。
没有设置基座轨迹、外力托举、修改机器人质量、缩小碰撞模型或取消障碍碰撞。

**这是仿真真值导航基线。** 规划读取完整赛道和障碍坐标，控制读取 MuJoCo 基座位姿；
相机已安装并可渲染 RGB／深度，但图像尚未用于导航。当前没有 MPPI、SLAM、ROS2 或视觉终点识别。
此阶段的作用是验证现有 loco 能否执行赛道要求，并给后续视觉算法提供可对照的基线。

**原视频在最后一个箱体旁存在严重的步态采样混叠。** 01 起点仿真 93–95 s，
同一只脚的触地周期约 0.31 s，而原视频以 3 倍速、10 fps 导出，每帧跨过 0.30 s，
几乎只展示相同的步态相位。两脚最低点的最大离地高度分别仅约 1.23、1.31 cm，
进一步强化了滑行观感。原速 50 fps 回放可见交替迈步。

同时存在短暂落脚滑移：每步接触点切向速度积分约 5 mm，大部分发生在触地后 50 ms 内；
稳定支撑段（同一足连续接触超过 50 ms）的按法向力加权滑移速度约 0.0022 m/s。
含触地瞬间的全接触均值约 0.057 m/s，不能将其解释成整个支撑期持续滑行。
只改变 NoSlip 迭代数 0→3 的短对照没有降低全接触均值，不能据此归因于软接触参数。
以下完赛结果仅表示通过当前路径判罚；落脚质量、扰动鲁棒性尚未完成验收。

## 运行

本机环境已经安装好，可直接运行：

```sh
# 默认：01 起点，包含三个障碍，完整身体包络规划，无图形窗口
.venv/bin/python scripts/run_sim.py

# macOS 实时观察；关闭窗口结束。Linux 也可用 python 加 --viewer
.venv/bin/mjpython scripts/run_sim.py --viewer --start 01

# 换起点、指定结果目录
.venv/bin/python scripts/run_sim.py --start 02 --output output/sim/start_02

# 从已经记录的物理状态生成原速 50 fps 视频，右下角显示机载相机
.venv/bin/python scripts/render_replay.py output/sim/start_01

# 重放相同指令检查实际接触滑移，并以原记录频率导出最后一个箱子片段
.venv/bin/python scripts/diagnose_slip.py
.venv/bin/python scripts/render_slip_audit.py

# 去掉障碍的巡线对照，以及错误的小包络对照
.venv/bin/python scripts/run_sim.py --no-obstacles --output output/sim/clear_track
.venv/bin/python scripts/run_sim.py --planner feet-only --output output/sim/feet_only

# 功能检查与三个官方策略的速度响应测量
.venv/bin/python scripts/test_sim.py
.venv/bin/python scripts/benchmark_loco.py
```

完赛退出码为 `0`；碰撞、摔倒、出界、时限结束或手动关窗返回 `2`。
默认时限为 240 秒，`--duration` 最大 300 秒。`--speed` 仅控制普通路段，障碍区由控制器降低速度。
交互窗口尚未进行人工操作验收；已经实际验证的是无窗口物理仿真和离屏渲染。

新环境：

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-sim.txt
.venv/bin/python scripts/fetch_pi_plus.py
.venv/bin/python scripts/generate_field.py
.venv/bin/python scripts/test_sim.py
.venv/bin/python scripts/run_sim.py
```

`requirements-sim-tested.txt` 记录本次测试的完整依赖版本。
`upstream.lock.json` 固定官方提交和每个文件的 Git blob SHA；下载器逐文件校验。
下载内容位于忽略目录 `vendor/mini_pi_plus_amp/`，包括模型、三个权重、原部署脚本、训练配置及许可证。
不需要安装 Isaac Lab，不在本机训练策略。

## 已验证结果

在固定地图、无额外扰动、无感知误差条件下，三个起点各测试一轮。
以下结果来自最终的连续线段碰撞检查版本，不是早期调参记录。

| 起点 | 一圈用时 | 过线后检查 | 停车期间位移 | 判罚 |
|---|---:|---:|---:|---|
| 01 | 120.62 s | 3 s | 0.111 m | 无 |
| 02 | 120.04 s | 3 s | 0.111 m | 无 |
| 03 | 120.66 s | 3 s | 0.109 m | 无 |

模型车道中心长度约 21.5635 m；其内圈与规则文字 19 m 的差异仍保留，见原 README。
三个起点的详细结果分别位于 `output/sim/start_01/`、`start_02/`、`start_03/`。
5 项回归检查覆盖真实行走、关节映射、单脚／双脚越界区别、障碍碰撞、摔倒及不可容纳包络。

**允许部分脚掌越线是这版成功结果的重要条件。** 判罚使用两只脚的实际碰撞网格投影：
两只脚均与跑道无交集才终止。足部投影不完全在线内的累计时长分别为 17.44、17.58、18.34 秒。
这不是“双脚始终完全在跑道内”的验证结果；若现场要求更严格，需要改变步态／手臂动作或路径约束。

基于本地场地和原始权重的失败对照也保留：小包络路径会造成手腕碰箱；
只把前进指令降低到 0.04–0.10 m/s 会出现明显死区。`model_40000.pt` 的 5 秒直行试验中，
0.08 m/s 指令只产生约 0.00063 m/s 的平均前向位移；0.12 m/s 指令约为 0.094 m/s。
0.12 m/s 纯侧移试验还带有约 0.019 m/s 的前向漂移。
这些是包含启动过程的平均值，不能视为实机稳态模型。完整测量在 `output/sim/loco_benchmark.json`。

## 实现和文件

| 文件 | 职责 |
|---|---|
| `sim/robot.py` | 合并机器人／场地，加载 AMP，构造观测、映射关节、执行 PD |
| `sim/navigation.py` | 已知地图路径格点搜索、身体包络检查、位置反馈和速度平滑 |
| `sim/scoring.py` | 接触、姿态、脚掌投影和越界检查 |
| `scripts/run_sim.py` | 时序调度、起点、终点停车、日志与完整状态记录 |
| `scripts/render_replay.py` | 已记录物理状态回放、机载画面、深度样本、轨迹图 |
| `scripts/benchmark_loco.py` | 对三个官方权重进行速度响应测量 |
| `scripts/test_sim.py` | 有物理执行的功能回归 |

底层使用 2 ms 物理步长、10 倍降采样，即 50 Hz 策略更新；每帧观测 69 维，5 帧历史，
20 维动作，动作缩放 0.25，观测顺序、关节顺序、默认姿态和 PD 增益与上游部署契约对齐。
策略用 CPU 推理。训练 checkpoint 通过 `weights_only=True` 加载；训练用 Normalizer 对象
以不执行代码的空状态类接收，actor 不使用其状态。未导入上游游戏手柄、键盘监听或 Isaac 环境。

上层当前也以 50 Hz 更新。普通路段跟踪中心线；障碍区保持身体朝向跑道切向，
用侧移绕行。规划采用 26 × 42 cm 的近似矩形身体包络，纵向余量 5 mm、横向余量 15 mm，
基座允许偏离中心线最多 23 cm。连续线段／扩张矩形的相交检查保证该近似路径不切过箱体；
实际是否碰撞仍以机器人原始三维碰撞网格为准。此矩形规划只支持当前水平直道上的轴对齐箱体，
改变该场景假设会明确报错。它不是适用于任意地形的通用全身规划器。

障碍区低速反馈加入了约 0.12 m/s 的平移指令下限来跨过当前策略的死区，仍保留加速度限制。
这个参数依赖当前权重；替换策略后应重跑速度测量和整圈测试。

每次运行输出：

- `summary.json`：成败、用时、原因、足部越线时长、姿态指标、规划假设。
- `telemetry.csv`：时间、真实位姿、速度指令、传感器速度、整圈进度。
- `states.npz`：每个策略周期的完整 `qpos` 与时间，用于物理结果回放。
- `planned_path.csv`：规划路径；`scene.xml`：包含机器人和场地的实际模型。
- 渲染后另有 `replay.mp4`、`trajectory.png`、起点／障碍相机图、米制深度 `.npy`。

生成的运行 `scene.xml` 使用本机资产绝对路径；跨机器迁移时应在新位置重新运行生成。
视频通过播放已记录的动力学状态生成，视频上的相机画面不参与控制。
`output/sim/slip_audit/` 保存滑移审计和 1 倍速 50 fps 局部回放；
审计重放与原记录的最大 qpos 误差为 0。接触力与速度在复制的状态上用 `mj_forward`
同步计算，不改变原仿真；统计只包含法向力大于 1 N 的足地接触，滑移阈值不是官方规则。

## 判罚和后续边界

机器人／箱体实际接触和非足部接地以 500 Hz 检查；脚掌凸包的地面投影以 50 Hz 检查。
摔倒还包含基座高度低于 20 cm 或倾角超过 60° 的工程阈值。
跑道区域取两条白线中心之间，终点取相对抽签起点前进一圈的中心线里程；
这些实现细节不是官方裁判算法。碰撞、摔倒、双脚出界任一发生会立即记录失败。

相机假设安装在基座坐标 `(0.09, 0, 0.22) m`，向下俯视 30°、垂直视场 70°。
当前 macOS 离屏深度渲染提示缺少 `ARB_clip_control`，深度精度受限；尚未据此验证深度测量误差。

接下来接入 RGB 白线／终点、深度箱体感知，用局部观测替换已知地图，并注入姿态误差、延迟、
摩擦和障碍位置变化。全视觉闭环、随机扰动成功率、实机部署和正式规则资格均未验证。

## 来源

官方仓库：[HighTorque Mini-Pi-Plus AMP](https://github.com/HighTorque-Robotics/Mini-Pi-Plus_AMP)。
机器人资产固定提交 `6a71f08177e0488122264250dbd1bfd4af70097f`，历史基线权重为 `Exported_policy/model_40000.pt`。
历史部署参数依据该版本 `legged_lab/scripts/sim2sim_pi_plus.py`；当前运控来源见本文顶部。
完整上游版权声明和 BSD-3-Clause
许可证保留在下载目录。模型、权重、媒体等第三方资产的再分发条款以其上游说明为准。

本次环境为 macOS、Python 3.13.5、MuJoCo 3.17.0、PyTorch 2.15.0、NumPy 2.5.4。

## RT11 箱区仿真

本地候选位于 `releases/shared_nav_20261111_rt11/`，基于 2026-11-11 20:17 的设备版本。每次仿真都从该目录复制控制器并记录源码哈希。MuJoCo 生成每一帧 RGB-D、IMU 和关节观测，使用 RT10 真机记录的相机、规划和速度发布时间间隔。裁判按机器人全身越过每个箱子的时刻记分。

从仓库根目录运行。macOS 使用 Python 3.11 环境中的 `mjpython` 打开实时三维窗口，右侧同步显示机载彩色图和深度图。

```sh
.venv-shared/bin/mjpython scripts/rt11_sim/run_validation.py full_01 \
  --source releases/shared_nav_20261111_rt11 --gui --hold \
  --out "output/sim/rt11_$(date +%Y%m%d_%H%M%S)"
```

`full_01` 从标准起点跑整圈。`boxes_center` 和 `boxes_oblique` 分别从箱区前方正向、斜向起步。无窗口验证可用同一环境的 `python` 并去掉 `--gui --hold`。新环境需先安装 `requirements-shared-sim.txt` 并运行 `python scripts/generate_field.py --bend-profile corner` 生成场地。

`summary.json` 记录全身过箱、碰撞和终止原因。`step_audit.json` 记录箱区实际发出的每步指令、零速回执、新相机帧以及白漆或深度的确认时间。`states.npz` 和 `camera_rgbd.zip` 可按原速同步回放。

```sh
.venv-shared/bin/mjpython scripts/rt11_sim/replay.py output/deployment/rt11_boundary_20261111/validated_full_01
```

回放窗口中按 `B` 跳到箱区、`P` 暂停或继续、`J` 和 `L` 前后跳 2 秒、`R` 回到起点。

本次使用同一份 RT11 控制器、0.22 m/s 的巡线设置。相机、关节与 IMU 使用噪声种子 1。三个启动场景均按全身接触和边界裁判计分。

| 起步方式 | 完成时间 | 箱区实际动作（横移 / 前进 / 转向） | 结果 |
| --- | ---: | ---: | --- |
| 标准整圈 `full_01` | 110.16 s | 31 / 19 / 2 | 依次越过三箱，接回巡线、走出小弯并完成一圈 |
| 正向箱前 `boxes_center` | 60.16 s | 31 / 20 / 1 | 越过三箱并驶出小弯 |
| 斜向箱前 `boxes_oblique` | 49.86 s | 28 / 15 / 1 | 越过三箱并驶出小弯 |

原始记录在 `output/deployment/rt11_boundary_20261111/validated_*`。标准整圈的三次全身过箱分别发生在 67.82、78.36、88.14 秒。三个场景的动作记录都确认了相邻两步之间实际下发零速、有停后的新帧，每次横移前看到了目标白漆，每次前进前有深度通行证据。整圈结束按裁判的几何过线计分，本轮未验证视觉终点停机。

额外用种子 2 跑同一标准整圈时，69.96 秒右腕碰到第二箱；详情见 `check_noise2/summary.json`。另外在首个横移期间注入 4 秒失线，最后一次非零请求在故障开始后约 0.10 秒，1 秒后下发的速度始终为零。以上为 11 月 12 日首次上机前的仿真记录。

11 月 12 日 11:38，RT11 真机发布目录更新了偏蓝白线识别。对现场运行 `20261112_032403Z_821e59d1` 的停步画面重算，卡住阶段可与 IMU 和遥测对齐的 60 组 RGB-D 中，旧版在 0 组确认目标白线，新版在 48 组确认。5 对满足时间间隔的画面都使新版给出下一小步横移。修复版重新通过上表三个标准仿真场景，真机安装时通过 29 项 RT11 测试与 8 项白线测试。回放与发布记录在 `output/deployment/rt11_tape_20261112/`。

11 月 12 日 11:52，RT11 更新了手脚位置与相机采集时间的匹配，并加强了近处浅蓝胶带的识别。现场运行 `20261112_033928Z_2c2bfcdf` 已进入第一箱沿线前进阶段，实际发出两次短前进，之后持续等待深度净空。使用第 13、14 次停车参考图离线重算，两帧均确认了 6.3 cm，中央深度缺失时均停车。当时未记录逐帧全身关节角，回放分别采用仿真步态包络和事后实测静止姿态。新版本再次通过上表三个标准闭环场景，机上通过 35 项 RT11 与 8 项白线测试。现场回放、仿真与部署回执在 `output/deployment/rt11_forward_20261112/`。

11 月 12 日 12:08，RT11 更新了第一箱前的横移重试。现场运行 `20261112_035448Z_419c9766` 中，底层收到约 0.14 秒右移，找线转身后连续 69 帧都在等待第一次位移证明。使用同一段现场遥测在 Jetson 离线重放，新控制器于 14.10 秒根据停步后的新观测给出一次右移。在继续输入静止测量的情况下，后续不再横移。三个标准仿真均过三箱并驶出小弯，额外整圈噪声种子 2 仍在第二箱发生右腕接触。真机安装通过 37 项 RT11 测试和 8 项白线测试，记录在 `output/deployment/rt11_lateral_20261112/`。

11 月 12 日 12:27，RT11 更新了箱区横移和前进。每步收步后用新画面决定方向，上一脚几乎没动时延长下一次同向指令。用 035448 的 17.66 秒停车记录在 Jetson 上离线回放，控制器发出 16 次横移，单步有效指令时长中位数 0.34 秒。白线对齐、前方有净空且机身位置固定的对照回放发出 15 次前进。正入和斜入三箱仿真均驶出小弯，标准整圈 110.16 秒完成。更换噪声种子后，第二箱仍有右腕接触。机上 50 项 RT11 测试和 8 项白线测试通过，安装后尚未进行现场试跑。[仿真与回放记录](output/deployment/rt11_continuing_20261112/validation.json) [真机安装记录](output/deployment/rt11_continuing_20261112/on_robot/robot_publish.json)
