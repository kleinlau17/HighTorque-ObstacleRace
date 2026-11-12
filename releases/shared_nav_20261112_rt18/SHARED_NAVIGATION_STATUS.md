# 共享导航与固定真机 AMP：2026-11-11

首版共享导航在旧 Mini 机身固定相机模型上通过 12 个物理边界场景、13 个传感器场景和 115.90 s 三箱整圈回归。当前模型使用 Pro 全身和双轴头部，成绩见下方 Pro 头姿验证。
本轮发现、修复和适用边界记录在 [边界场景测试](CORNER_CASES.md)。

## 地面纹理关键帧建图实验

从 `0a40647` 的共享导航开始。原有白线点集和箱体跟踪用于局部规划，白线、箱面、箱顶轮廓配准及相邻地面图像匹配提供短时平移，箱区可用已观测的走廊和箱边重定位。单条直线沿线方向的位移存在歧义。近处地毯有二维纹理时，前后两次拍到的同一片地面可同时约束纵向和横向位移。

`navigation/floor_map.py` 把已通过地面深度检验的像素按 AHRS 姿态投到水平面。每张图覆盖前后 40 cm、左右各 10 cm，分辨率为 2 mm。在第一帧有效地面上，从机身前方 45 cm 起以 2 cm 步长选取窗口。优先选取当前姿态与机身回正时都留有 5% 图像边距、且每个插值源像素均有地面深度支持的最近窗口。选定后整圈固定在该机身相对距离，以便比较不同时间的纹理。沿途约每移动 10 cm 或改变朝向 0.12 rad 保存一张，最多 256 张，纹理阵列约占 19.5 MiB。仿真与机载共用 `NavigationRuntime`，每次运行从空图开始。

- 两个以上不重叠的地面纹理块须给出一致的二维平移；每块均检查双方向梯度、归一化相关峰值和次峰间隔。邻近关键帧跟踪要求两次测得的位置满足步行速度范围；重访和失跟恢复要求两次修正量相差不超过 2.5 cm。匹配使用现有的 [OpenCV 模板相关](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html)。
- 正常跟踪时将里程计与邻近关键帧配准。累计行程相隔至少 3 m 时，允许在地图估计位置周围 1 m 搜索方向相近的旧关键帧。重访只修正地图平移，短程跟踪器继续用自己的位姿、箱体记忆和侧移目标。坐标设计参考 [ROS REP 105](https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst)。
- 局部跟踪中断超过 0.25 s 时，可用同一里程计段内近期关键帧重获最多 18 cm 的位移。观测确认后重新初始化局部跟踪，累计三帧可信视觉再恢复控制。局部里程计盲重启后保留旧图，`map_xy` 暂为 `null`，识别到旧图的位置后才继续融合。
- 机载和仿真在运行目录输出 `floor_map.npz`，包含每个关键帧的位置、方向、时间戳、段号、各帧使用的光心和光学旋转、相机内参、取样窗口原点和深度校验的高通纹理。遥测中的 `pose` 是局部控制位姿，`map_xy` 是相同起始坐标下的地图位置。地图只估计平移，航向来自 AHRS；历史关键帧保留采集时的估计，未做全局位姿图优化。

### Pro 任务头姿下的建图验证

仿真按照 261010 Pro 的 `camera_link` 和实测头部反馈建立机身到光学坐标变换。任务俯仰反馈为 0.63837165 rad，偏航为 0 rad，实机恢复时的俯仰保持命令为 0.61046015 rad。复现 standby 机身横滚 −1.26°、俯仰 +5.02° 时，渲染深度独立拟合出相机对地俯角 43.95°，对应实测 43.97°。进入零速 AMP 站立后，仿真深度拟合得到约 39.00°。建图在每张 RGB-D 图像的时间点应用 AHRS 姿态。[站姿核对](output/sim/pro_calibrated/standing/standing.json)

机载旁路使用同一参考头姿下的名义外参，按图像接收时间取最近的两轴反馈。反馈超时、缺轴或头部偏离参考时跳过该帧，原因记录在 `commands.json`。镜头平移取自 CAD 估计，单个地面法向无法确定全部外参，IMU 对齐仍待实测。对存档的实机静态 RGB、对齐深度和同次 IMU 离线检查，选中的纹理块从前方 53 cm 起，所有采样通过地面深度约束，地面拟合 RMS 为 1.88 mm。这份记录仅覆盖一个静态视角。[实机存档检查](output/integration/pro_head_mapping/real_frame_final.json)

从 01 起点按实测 standby 姿态初始化，保留固定原厂 AMP，速度目标为 0.22 m/s，启用种子 1 的传感器噪声。偏航对照将整个机身绕场地竖轴旋转 0.14 rad（约 8°），头部偏航维持 0。独立裁判给出赛道进度和碰撞事件，地图位置由同步物理状态事后评分。

| Pro 仿真场景 | 赛道结果 | 地图位置 RMSE | 地图记录 |
|---|---|---:|---|
| 正向起步，三箱 | 净进度约 1 cm，17.78 s 后因停滞结束 | 0.11 cm | 266/267 帧接受里程计更新，1 张关键帧，未重置里程计段。[评估](output/integration/pro_head_mapping/official/mapping_evaluation.json) |
| 机身偏航 8°，三箱 | 15.69 m（72.7%），61.58 s 左腕碰 02 号箱 | 0.66 cm | 101 张关键帧，箱体附近 RMSE 0.97 cm。[定位](output/integration/pro_head_mapping/boxes/mapping_evaluation.json) · [裁判](output/integration/pro_head_mapping/boxes/evaluation.json) |
| 机身偏航 8°，无箱 | 79.24 s 实际过线，79.30 s 视觉确认，继续观察 3 s | 0.63 cm | 144 张关键帧，13 次确认重访更新，最大误差 1.77 cm。[地图预览](output/integration/pro_head_mapping/lane/floor_map.png) · [定位](output/integration/pro_head_mapping/lane/mapping_evaluation.json) · [裁判](output/integration/pro_head_mapping/lane/evaluation.json) |

上述闭环均选中了前方 49–89 cm、宽 20 cm 的地面窗口。将保存的各关键帧投影参数与同一时间的 Pro 头部正向运动学逐帧比较，两次行走试验最大光心偏差小于 0.012 mm、最大旋转偏差小于 0.007°。这验证了仿真渲染与建图使用的变换一致。正式三箱赛道仍有起步和绕箱通行问题。

另取已完成三箱整圈的历史机身及身体关节轨迹，用当前 Pro 场景、固定任务头姿和 D435i 重新生成含噪 RGB-D 和 AHRS。位置只用于摆放渲染模型和事后评分。这次验证覆盖 1,783 帧、172 张关键帧，地图 RMSE 为 2.00 cm、最大误差 3.41 cm，15 次重访更新。该试验复用规定轨迹，未执行 Pro 步态闭环；箱区被遮挡的地面在纹理地图中留空。[三箱轨迹地图](output/integration/pro_head_mapping/three_box_path/floor_map.png) · [评估](output/integration/pro_head_mapping/three_box_path/mapping_evaluation.json)

固定依赖的 Python 3.8 重放无箱整圈的 4,115 个控制周期、1,235 帧图像，地图位置、匹配计数和速度请求与 Python 3.11 闭环逐帧一致。桌面回放相机周期 P95 为 74.2 ms，Jetson 实时性待实测。Python 3.8 独立部署包的 179 项测试通过；Python 3.11 全仓 227 项中 226 项通过。唯一失败是原厂 AMP 正向起步回归，直行请求 3 s 后基座仅前进 2.8 cm，低于 35 cm 的门槛。[回放核对](output/integration/pro_head_mapping/replay_lane.json) · [部署包测试](output/integration/pro_head_mapping/verified_package38.log) · [全仓测试](output/integration/pro_head_mapping/verified_tests311.log)

### 旧模型基线：机身固定相机的三箱闭环与定位误差

01 起点，D435i 默认配置，固定 Pro AMP，噪声种子 1 与 2 各跑一圈。独立裁判核验了完整进度、实际过线、碰箱、摔倒及双脚出界，过线后继续观察 3 s。以下位置误差按相机时间对齐物理状态，仅使用已知起步位姿转换评估坐标，无事后轨迹拟合。RMSE 为二维位置误差的均方根。

| 噪声种子 | 完赛 | 局部里程计 RMSE | 地图 RMSE | 地图最大误差 | 停车末帧地图误差 | 箱体附近 RMSE，局部 → 地图 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 115.90 s | 12.81 cm | **0.86 cm** | 4.83 cm | 1.32 mm | 2.86 → 0.39 cm |
| 2 | 113.22 s | 18.45 cm | **5.40 cm** | 40.80 cm | 0.65 mm | 2.17 → 1.21 cm |

两个种子均无裁判事件，最后 1 s 三轴请求全零。与首版同种子对照，逐周期速度请求和物理轨迹一致。默认整圈记录了 174 张关键帧、429 次邻近关键帧位置更新，115.28 s 首次确认重访。第二个种子在 111.34 s 出现最大误差，首次确认重访为 112.94 s。终点处的小误差不能代表全程精度。箱体附近指前后 −0.40 至 0.85 m、左右 0.80 m 内有已跟踪箱体的相机帧，共 320 和 323 帧。

[默认种子地图预览](output/integration/visual_mapping/full_lap/floor_map.png) · [整圈回放](output/integration/visual_mapping/full_lap/replay_1x_50fps.mp4) · [默认种子定位评估](output/integration/visual_mapping/full_lap/mapping_evaluation.json) · [第二种子定位评估](output/integration/visual_mapping/lap_seed2/mapping_evaluation.json)

### 旧模型退化场景与部署边界

- 12 个物理边界试验均通过原验收条件，覆盖起步与大小弯单边线、箱区单边线、直道及箱区 RGB-D 中断、深度孔洞和偏航起步。直道断图、深度孔洞触发的分段保留了中断前关键帧，恢复画面后重新确认地图位置。[物理场景评估](output/integration/visual_mapping/physical_cases/corner_results.json)
- 14 个透视传感器试验中，12 个按预期恢复，另外两个保持停车。箱体遮挡且相机断帧 0.6 s、相对最后有效帧实际移动 11.2 cm 时，首版持续等待，新版经两帧确认位移，画面恢复后 0.38 s 首次给出非零请求；首次重获的位置误差为 0.036 mm。这个数值来自解析相机。无纹理，以及实际移动 21.2 cm 的超范围对照均保持停车。[传感器故障结果](output/integration/visual_mapping/sensor_cases/sensor_recovery_summary.json)
- 本机读取原厂快照逐个核对了 8 个文件的 SHA-256，其中引擎、AMP 和 PD 配置同时匹配冻结契约。控制器二进制身份来自前次实机审计。仿真使用上游 Mini Pi Plus 动力学和从 Pro 引擎导出的同组权重。[AMP 身份记录](output/integration/visual_mapping/amp_identity.json)
- 地毯纹理、可用的米制深度、相机到机身的外参和 AHRS 方向均会影响匹配。地图覆盖实际可见的地面条带。没有可信运动观测时仍会停车；地图没有跨任务载入、任意位置初始化或全局航向校正。真机未连接，Jetson 延迟、现场地毯、真实遮挡和制动距离仍待测。

### 旧模型代码与回放核验

本机 Python 3.11 共 207 项单测通过，Python 3.8 导航和接口测试共 179 项通过。独立临时打包后，Python 3.8 再跑 13 项建图及传感器恢复测试通过。固定 NumPy 1.24.4 和 OpenCV 4.10.0。两个 Python 版本重放同一圈的 5,945 个控制周期、1,784 帧 RGB-D，逐帧地图坐标、匹配计数和速度请求与原始记录一致。

两份回放并发运行时，含规划及逐帧核对的相机周期 P95 分别为 137.7 ms（3.11）和 137.6 ms（3.8），超过 15 Hz 的 66.7 ms 帧周期。另用同一圈的有效观测单进程测量，建图阶段 P95 为 8.3 ms。这些桌面数据用于计算一致性与开销评估；Jetson 的实时吞吐仍需旁路实测。[汇总与文件身份](output/integration/visual_mapping/validation.json) · [建图阶段计时](output/integration/visual_mapping/time_mapping.json)

### 仿真与地图复现

```sh
export OPENBLAS_NUM_THREADS=1
run=output/sim/mapping_pro_boxes
.venv-shared/bin/python scripts/run_visual.py --start 01 --spawn .55 -1.25 .14 --duration 140 --video --record-sensors 100 --output "$run"
# 无箱整圈对照使用同一初始姿态，并写入独立目录
.venv-shared/bin/python scripts/run_visual.py --start 01 --spawn .55 -1.25 .14 --no-obstacles --duration 140 --output output/sim/mapping_pro_lane
.venv-shared/bin/python scripts/evaluate_visual.py "$run"
.venv-shared/bin/python scripts/evaluate_floor_map.py "$run"
.venv-shared/bin/python scripts/replay_navigation.py "$run/sensor_replay.npz" --map-observations "$run/observations.json"
.venv-shared/bin/python scripts/reproject_floor_lap.py --trajectory output/integration/visual_mapping/full_lap \
    --scene "$run" --output output/sim/mapping_pro_prescribed
.venv-shared/bin/python scripts/run_corner_cases.py --output output/sim/mapping_corners
.venv-shared/bin/python scripts/test_sensor_recovery.py --output output/sim/mapping_sensors
```

## 首版记录

以下记录此前 `full_06` 与 `shared_nav_20261111_02` 包的验证结果。
该版本已在带红、黄、绿三个箱子的仿真赛道完成一圈，并通过独立过线与停车核验。
最终运行是 `output/integration/obstacle_fix_20261111/full_06`，01 起点，D435i 成像参数，传感器噪声开启。
这次没有连接或操作真机；Pro 动力学、相机外参和真实传感器延迟仍待实机验证。

## 现在运行的方案

```text
RGB + 米制深度 + AHRS 姿态
  → 深度平面估计地面，识别白线、终点和箱体
  → 白线/箱面配准 + 地面纹理匹配，估计短时运动
  → 局部障碍记忆 + 横向网格动态规划
  → 路径跟踪 / 箱间减速与侧移闭环，生成 vx、vy、wz
  → 原厂速度滤波、死区适配
  → 固定真机 AMP → 关节控制
```

仿真 `scripts/run_visual.py`、离线回放 `scripts/replay_navigation.py`、机载 `deploy/shared_navigation.py`
都调用 `navigation/runtime.py` 的 `NavigationRuntime`。导航没有世界位姿、世界速度、预设跑道或箱子坐标、
物体分割、真值接触信息接口；世界真值只供独立裁判与事后分析使用。规划器是局部动态规划，不是 MPPI。
原 `deploy/line_follow_core.py` 保留为历史巡线实验，不参与共享导航路径。

基础运控固定为这台 Pro 的 `amp.trt`，配置名 `walk/pi_plus_amp_1110_policy_1.rknn`。
权重、69 维单帧观测与五帧历史、关节顺序、动作缩放、50 Hz 策略、1000 Hz PD、原厂速度滤波与死区均未改动。
仿真使用同一组导出的 FP32 权重，和已保存的 Jetson 引擎输出及真实运行观测/动作记录做数值比对。
这保证运控计算对应，不代表上游 Mini Pi Plus 的物理模型已等同于 Pro 实机。

## 本轮修复

1. **三个箱子分别为红、黄、绿。** 只修改视觉颜色；位置、20 × 20 × 40 cm 尺寸、碰撞盒、跑道和摩擦不变。
   [颜色预览](output/integration/obstacle_fix_20261111/three_color_scene.png)。
2. **地面高度不再由绿色像素决定。** 使用深度点云、重力方向、平面残差和连通面积判断支撑地面，排除箱顶。
   当前深度无法支持地面估计时明确判无效，不用旧高度冒充新观测。
   红黄绿、全绿（与地毯完全同色）、全白等输入回归得到一致的地面和箱体几何结果。
3. **修复遮挡期间的箱体尺寸和位置跳动。** 已知箱子保留测得的方向与遮挡记忆；内部木板碎片不能覆盖外盒尺寸。
   真正旋转 45° 的 20 cm 方箱仍按约 28.3 cm 投影宽度检查，不把它硬缩为 20 cm。
4. **修复视觉定位失效与错误累计。** 配准区分白线、箱面和箱顶轮廓，拒绝不可观测方向；
   纹理不足的直线场景使用深度校验后的米制地面图像相关匹配。全部插值来源必须有有效地面深度。
   连续地面跟踪之后，新几何先作候选，下一帧配准确认后才接回几何参考；弱稀疏光流不能更新几何锚点。
   不积分下发速度，不使用仿真专用关节/接触里程计。
5. **修复窄箱间隙内的跟踪。** 用观测到的直道方向作为侧移参考，先减速、再闭环侧移；
   共同位置修正同步到记忆中的箱子和侧移目标。实际不移动时有限次站立恢复，仍不移动则停车。
6. **修复终点晚判。** 终点仍在视野内时持续更新测得位置，离开视野后由短时视觉里程计判断过线。
   过线法向不随机器人原地转身反向，也不用计时或指令积分预测完成。

## 最终闭环结果

| 项目 | 实测结果 |
|---|---|
| 自主完成时间 | **111.56 s** |
| 独立裁判确认机身实际过线 | 111.54 s |
| 完成触发时距终点线 | 前方 6.7 mm；横向距线中点 11.9 mm |
| 碰箱 / 摔倒 / 双脚均离开跑道 | 均未发生 |
| 完成后观察 | 3.00 s |
| 停车阶段总位移 | 6.27 cm |
| 最后 1 s 位移与指令 | 0.177 mm；三轴原始请求均为 0 |
| 独立整圈核验 | 通过；裁判进度为中心线一圈的 100.32% |

接触在原始 1000 Hz 物理步中检查，脚与跑道重叠在 50 Hz 检查。允许单脚部分越线，
不能将该结果描述为双脚始终完全在线内。裁判没有放宽；事后验收另外增加终点位置检查，
拒绝“绕满一圈但远离终点才宣布完成”的结果。

- [整圈回放：1 倍速 / 50 fps](output/integration/obstacle_fix_20261111/full_06/replay_1x_50fps.mp4)
- [独立轨迹图](output/integration/obstacle_fix_20261111/full_06/trajectory.png)
- [独立评估数据](output/integration/obstacle_fix_20261111/full_06/evaluation.json)
- [完整汇总与文件身份](output/integration/obstacle_fix_20261111/validation.json)

之前的失败记录继续保留：`full_02` 箱间碰撞；`full_03` 终点前定位失效；
`full_04` 内部碎片误增箱宽导致阻塞；`full_05` 虽走完全程但终点晚判 66 cm，按更严格的终点检查不通过。
这些记录不作为最终版本通过证据。

## 测试与复现

- Python 3.11：117 项测试通过，含 AMP 数值对照、相机投影、共享导航、地面/颜色、定位恢复、绕障与终点。
- 本机 Python 3.8：87 项纯导航与接口测试通过。
- 完整 114.56 s 传感器记录（5,728 个控制周期、1,719 帧图像）在 Python 3.11 与 3.8 下重放，
  速度指令与仿真原始记录的最大差值均为 **0**，源文件哈希也逐项一致。
  报告：`full06_replay_python311.json`、`full06_replay_python38.json`，位于本轮结果目录。
  这验证本机跨 Python 计算一致性，不是 Jetson 实时性或真实相机检查。
- 日志：`output/integration/obstacle_fix_20261111/tests_release_final.log` 与 `tests_python38_final.log`。
- 固定 NumPy 1.24.4、OpenCV 4.10.0；入口拒绝依赖版本不符。
  本机旧 OpenCV 5 环境不能作为相同导航的验证环境。

```bash
uv venv --python 3.11 .venv-shared
uv pip install --python .venv-shared/bin/python -r requirements-shared-sim.txt
OPENBLAS_NUM_THREADS=1 .venv-shared/bin/python scripts/run_visual.py --duration 180 --video --record-sensors 150 --output output/sim/shared_three_colors
.venv-shared/bin/python scripts/evaluate_visual.py output/sim/shared_three_colors
.venv-shared/bin/python scripts/render_visual.py output/sim/shared_three_colors
```

## 部署边界

本轮验证的是上述单一起点、场景、噪声种子下的物理闭环与针对性回归，不是所有布局、光照和材质的可靠性保证。
箱体检测不使用颜色标签，但相机仍需有效深度，视觉定位仍需要可观测的几何或纹理。

模型仍采用上游 Mini Pi Plus 动力学；Pro 实机的惯量、碰撞、摩擦与相机安装尚未校准。
AHRS 是带噪姿态测量的仿真，选定起点的 3 cm 横向终点线是明确的场景假设。
跑道与箱子具体布置也保留 README 中注明的建模假设。

新部署包为 `shared_nav_20261111_02`；旧 `_01` 不覆盖。默认旁路，不创建运动发布者。
包内附带整圈开头 6 s 的轻量传感器样例；完整记录与两份整圈重放报告保留在本机结果目录。
包内外参和 IMU 标定标志仍为未验证，原厂文件只读校验。
连接真机后按 [部署步骤](deploy/SHARED_NAVIGATION.md) 完成依赖/哈希预检、同文件回放、真实传感器旁路、
头部/相机/IMU 标定与现场可接管的闭环验收。不能直接把占位标志改成真来跳过标定。
