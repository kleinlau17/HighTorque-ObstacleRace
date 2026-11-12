# 共享导航版本：部署与验收

本版本将 RGB-D 感知、视觉里程计、局部绕障规划、路径跟踪和 AMP 速度适配放到同一个 `navigation/` 包。
仿真入口 `scripts/run_visual.py`、离线回放 `scripts/replay_navigation.py` 和机载入口
`deploy/shared_navigation.py` 使用同一个 `NavigationRuntime`。`sim/visual_navigation.py` 只保留兼容导出。
旧 `line_follow_core.py` 与 `--calibrated-turns` 保留用于历史诊断，不是共享导航的执行路径。

## 当前版本的边界

- 冻结真机 `amp.yaml`、`amp.trt`、PD 参数和主控制器二进制的 SHA-256；启动前只读比对。
- 导航不推理另一套 AMP，也不覆盖原厂程序。仿真使用该 TensorRT 引擎提取的同一组 FP32 权重，已有数值对照。
- 视觉输入是 RGB、米制深度、相机内外参和 AHRS 姿态。无世界位置、已知跑道地图、箱体真值或仿真接触信息输入。
- 规划器是局部横向网格上的动态规划；跟踪器根据前方路径点、路径方向计算 `vx, vy, wz`，不是 MPPI。
- 地面由重力约束下的深度平面估计，不使用绿色像素定高度。箱体按相对地面高度、表面和尺寸识别，
  红、黄、绿分别作为同类障碍；已观测箱体的方向与遮挡记忆用于避免局部碎片误增尺寸。
- 短时定位优先配准白线/箱面；地面纹理以米制鸟瞰图作相关匹配，并检查深度、纹理可观测性和匹配一致性。
  箱体之间的窄通道采用观测跑道方向下的位置闭环，先减速再侧移；实际位移不足时允许有限次站立恢复。
- `AMPCommandAdapter` 在导航侧补偿原厂 0.3 滤波和死区，限制原始请求在 `±0.4, ±0.4, ±1.0`。
  小于策略实际起步能力的前进目标会升到 0.20 m/s；这是导航目标适配，不能把 `--speed` 理解为硬件实际速度保证。
- 原厂滤波后的速度量是策略输入，不是机身真实速度。视觉里程计才提供当前实现的平移估计。
- 视觉跟踪失败：保留最后可信参考帧和障碍记忆，输出零；三帧可信观测后恢复。
  不用速度积分或仿真专用的脚底里程计冒充实际位置。无箱体记忆、无终点目标时，停车后可重建局部参考段；重建次数单独记录，累计路程不是精确全局定位。
- 仿真使用 261010 Pro 全身模型和双轴头部。机器人接触和驱动响应、相机光心和 IMU 安装方向、相机与关节反馈的实际时延仍需实机标定。
- 包内 `pro_config.json` 的名义外参对应实测任务头姿（俯仰反馈 0.63837165 rad，偏航 0 rad）。旁路与运动入口都按 RGB-D 接收时间检查两轴反馈。反馈缺失、过期或偏离参考的图像跳过投影，原因见 `commands.json`。包内两个标定标志保留为 false，仅允许旁路。

导航依赖也已固定：NumPy 1.24.4、OpenCV 4.10.0（wheel 4.10.0.84）。
旧本机的 OpenCV 5 与目标环境会产生不同指令，不能混用；入口会拒绝版本不符的环境。

当前验证结果见项目根目录 `SHARED_NAVIGATION_STATUS.md`；旧策略的整圈成绩不能用于此版本验收。

## 明天连接真机后的顺序

1. 将不可变压缩包传到 `/home/nvidia/competition_nav/incoming/`，先核对包的 SHA-256。
   使用同一文件重传；传输回包不确定时先查哈希，不创建新的任务。
2. 解压到新的 `releases/<release_id>/`。旧版本、原厂目录、自启动都保留。
3. 在新目录执行以下命令（均不启动行走）：

```bash
bash deploy/install_shadow.sh
source /opt/ros/noetic/setup.bash
source /home/nvidia/sim2real/install/setup.bash
.venv/bin/python -m unittest scripts.test_shared_navigation
.venv/bin/python scripts/replay_navigation.py validation/sensor_replay.npz --output runs/replay.json
.venv/bin/python deploy/shared_navigation.py --preflight --output runs/preflight
.venv/bin/python deploy/shared_navigation.py --duration 20 --output runs/shadow
```

`install_shadow.sh` 只在本版本内建虚拟环境/相机采集程序；复用版本完全匹配且通过检查的原厂 NumPy/OpenCV，
否则只安装到本版本虚拟环境。不会启动或替换原厂控制器。
`--preflight` 检查代码清单、依赖与原厂身份；`shadow` 只订阅传感器并计算，不创建速度或关节发布者。

4. 对比 `preflight.json` 与仿真/本地回放的 `navigation_core_sha256`。
   回放要求 `maximum_command_difference < 1e-5`；Jetson 性能以真实 `camera_cycle_ms_p95` 为准。
   本地已创建 Python 3.8 环境做实际回放与单元测试；这仍不等于 Jetson/ROS/相机检查已经通过。
5. 核验头部反馈、相机内参、50 cm 跑道尺度、相机到机身外参、IMU 轴向和曝光/接收时延。
   保存独立的标定配置与证据；不能直接把占位配置中的两个 `verified` 开关改成真。
6. 标定通过后才进入运动验收：确认现场状态后，经 `--shared-navigation` 的监督器执行。
   标定配置需包含 `calibration_evidence`、`head_reference_rad`（两个头关节）、已验证的矩阵与位置。

机载回放、旁路、标定和受监督闭环未通过之前，不把仿真通过解释为真机可直接跑一圈。

## 真机终端启动整圈测试

安装后可以在真机终端或 SSH 中使用入口 `deploy/nav`。部署时将
`/home/nvidia/competition_nav/nav` 链接到当前版本的入口。

```bash
/home/nvidia/competition_nav/nav head
/home/nvidia/competition_nav/nav start
/home/nvidia/competition_nav/nav status
/home/nvidia/competition_nav/nav stop
```

- `head` 连续恢复俯仰和偏航，等待两轴反馈达标后显示 `head_ready`。偏航目标为 0 rad。
  俯仰保持命令为 0.61046015 rad，验收反馈为 0.63837165 rad。
- `start` 检查当前头姿、相机/IMU 标定和原厂 AMP running 状态，然后在机载后台启动整圈。
  已处于标定头姿时可以直接使用。最长 300 秒，通过视觉终点后提前结束。
  默认导航目标速度为 0.22 m/s，可用 `--speed` 指定，实际速度由策略响应决定。
- `status` 显示最近一次终端任务、PID、日志路径和结果。`stop` 请求结束任务并等待控制器清理。
  运行中重复 `start` 会显示任务占用，避免重复启动。行走任务独立于 SSH 会话运行。
- `check` 进行只读环境和身份核验，并列出相机/IMU 标定缺失项。

按用户 2026-11-11 指示，终端入口默认使用 `deploy/pro_model_config.json`，
`calibration_basis` 为 `robot_model_assumption`。相机位姿采用 Pro URDF 头部运动链与
`sim/pro_camera_mount.json` 中的光学安装估计，IMU 轴向采用仿真中与 `base_link` 对齐的设置。
配置保存模型来源、文件哈希和假设列表，实测验证字段仍为 `false`。
该显式模式允许按模型几何启动导航，矩阵有效性、当前头姿、传感器时效和控制接口继续参与检查。
可通过 `start --config /绝对路径/real_config.json` 或环境变量 `NAV_CALIBRATION_FILE` 使用后续实测配置。
头姿恢复本身可独立执行。终端入口复用原监督器的人工接管、速度超时和控制接口校验。
原厂 ROS 手柄翻译器仍属于速度输出链路，启动不依赖 Linux 游戏手柄设备或组合键事件。

### 机载实时性与时效窗口

2026-11-11 的机载 profile 显示单帧 `runtime.observe()` 约 1.01 s（33 帧 33.5 s），整帧周期
p50 1.43 s、p95 1.95 s。当时链路上的时效判据是 0.2–0.5 s，运行时因此每一帧都被判成
`stale_camera`／`lost_vision`，`good_frames` 只有 1，持续发布零速度：`nav start` 显示已启动，
机器人却站立不动。本次改动把这三处窗口统一到配置项 `observation_lease_s`（默认 1.0 s，
同时作用于运行时、导航器和 50 Hz 转发线程），去畸变直接输出到感知分辨率，相机按 6 fps 请求。
白线拟合的 43 条直线候选改为矩阵化求解并统一打分，抽样顺序与随机数消耗保持不变，
120 组随机点云与改动前逐位对比一致，桌面单次耗时从 1.61 ms 降到 0.75 ms。
`perception_scale`（0.25–0.5）决定感知分辨率，降低它以换取帧率。
每帧的 `remap_ms`／`processing_ms` 写入 `telemetry.jsonl` 和 `vision_summary.json`，
验收目标是 `processing_ms_p95` 不超过 400 ms。这些数值来自机载测量，尚未在真机整圈中复核。

## 手柄两次按键整圈测试

`navigation_buttons.py` 提供原厂手柄组合键入口。组合键默认留空，现场核对原厂动作后指定编号。
正常测试从原厂 AMP running 状态开始，操作顺序如下。

1. 第一次按住组合键 1 秒后全部松开，连续恢复 `task_head_pose.json` 中保存的头部视角。
   俯仰命令为 `0.61046015 rad`，验收反馈为 `0.63837165 rad`，偏航验收值为 `0 rad`。
   两个关节从当前反馈位置同步平滑恢复，偏航目标为 `0 rad`。
   已到位的关节使用保存的保持命令，例如单独恢复偏航时，俯仰保持 `0.61046015 rad` 的已标定命令。
   俯仰反馈误差需在 `±0.025 rad` 内，偏航反馈误差需在 `±0.04 rad` 内，才进入下一阶段。
2. 到位后显示 `head_ready`，等待第二次完整按住、松开。头部运动期间的组合键操作会被丢弃。
   在 standby 下也可以完成头姿恢复，随后提示通过原厂手柄进入 AMP running，再接受第二次操作。
3. 第二次操作启动共享视觉导航，识别并通过终点后结束，最长 300 秒。
   到时尚未识别通过终点记录为 `navigation_time_limit`。新手柄入口仅提供整圈模式。
4. 摇杆或其他原厂按键用于接管。头部准备期间的人工操控会取消准备，行走期间由原有监督器接管。
   每轮退出后，下一次组合键从头部准备重新开始，记录保存到独立的 `lap_*` 目录。

这里的头部步骤恢复并核验已有姿态，完整相机外参和 IMU 对齐仍使用独立实测配置。
程序接受 `--config` 指定的标定文件，并检查其头部参考值与保存姿态一致。

连接真机后先查看原厂按钮编号、控制状态和设备连接情况。

```bash
bash deploy/run_navigation_buttons.sh --inspect-buttons
```

现场检查候选组合键的单键、组合按下及释放是否触发原厂动作。
检测入口只读取信息，原厂按键通道仍直接接收手柄输入。
将核对后的编号（逗号分隔）放入 `NAV_START_BUTTONS`，将实际标定文件路径放入 `NAV_CALIBRATION_FILE`，启动监听。

```bash
bash deploy/run_navigation_buttons.sh \
  --start-buttons "$NAV_START_BUTTONS" \
  --config "$NAV_CALIBRATION_FILE" \
  --background
```

监听进程在机载后台运行，状态见 `runs/navigation_buttons/status.json`，
日志见同目录 `launcher.log`，每轮包含头部反馈报告与导航日志。
它通过原厂 `/joy_node/dev` 指定的 Linux 手柄设备检测断连，同时读取 `/joy` 上的按键。
检测到设备断开会取消手势、请求停止当前任务，重连后需重新按键。
设备路径可用 `--joy-device` 指定。该连接检测仍需在实机上核验。

本机已覆盖两次按键、长按去重、准备期间按键丢弃、人工取消、控制锁、头姿失败、
standby 等待、超时与视觉完赛的分别记录。具体组合键、设备断连行为和实机闭环仍待现场验证。

## 执行路径和保护

机载视觉进程仅发 `/competition/lane_request`，内容是期望策略速度。
独立 50 Hz 监督器读取策略观测的第 6–8 项，在实际发布前做同一份 AMP 速度适配。
只有原厂手柄翻译器的受控入口发布 `/cmd_vel`，手柄输入优先；视觉请求超过 0.2 秒失效。
监督器校验运行模式、AMP 身份、关节反馈、标定头部角度和输入数值的有效性，任务按 `--duration` 指定的时长结束。
电量、机身倾斜、实测转速和累计指令量仅写入日志。电池消息缺失或过期也仅记录。
`guard.jsonl` 的 `battery` 记录电量与消息年龄，`attitude` 记录倾角与转速。
旧日志键 `distance_command_budget`、`turn_command_budget` 继续记录指令积分，预算上限在任务摘要中为 `null`。
监督器使用跨版本共享锁，`launch_line_trial.py` 使用任务目录标记防止断网重试重复启动。
退出后恢复原厂手柄翻译器；导航停止后实际脚步仍有停车距离，不能把零请求当作瞬时静止。

该过程完全在机载运行；SSH 的延迟只影响取日志和管理，不放宽传感器新鲜度要求。
累计指令量由原始速度请求积分得到，实测行走距离由视觉里程计提供。

## 旁路输出

- `preflight.json`：Python/NumPy/OpenCV、文件哈希与标定抑制原因。
- `floor_map.npz` 保存本次运行的米制地面纹理关键帧、实际取样窗口原点、相机内参和每张关键帧采用的机身相对光心与光学旋转。遥测中的 `map_xy` 为地图坐标；里程计分段后等待重定位时可为 `null`。真机上需测量实际地毯的纹理可辨性和包含配准的循环用时。
- `camera_intrinsics.json`、`config.json`：本次成像参数和标定输入。
- `telemetry.jsonl`：视觉观测与定位状态；`processing_ms` 仅为感知用时。
- `commands.json`：候选原始速度、期望策略速度、抑制原因、包含规划的循环用时。
- `summary.json`：旁路发布者数量、包含规划的 `camera_cycle_ms_p95`、最终状态。
- `latest_rgb.jpg`、`latest_overlay.jpg`：当前相机画面与感知结果。

标定未验证时，旁路速度只是诊断候选，不能作为运动依据。
