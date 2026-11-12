# RT11 真机短步测试版

箱区默认横移一小步最多 4.6 cm，前进一小步最多 6.3 cm。这两个数是控制器的估计值。每步结束时先发零速，再看收步后的新画面。对侧白线还偏着，就继续横移。白线对齐后向前迈步。深度相机只看不全脚边、已对齐的白线仍可见时，就沿白线前进。上一脚几乎没动时，控制器会延长下一脚同方向的指令，继续尝试。箱子或白线一时看不清时，会短暂观察，再小幅转身换个角度看。

# RT11 三箱导航

箱区控制从 RGB-D 的箱面和顶缘估计箱体方向与相对位置，用足底运动学和 IMU 在视觉帧之间修正前进、横移和制动指令。每个发布周期使用已完成的一帧和与之配对的控制目标。箱区观察超时或 IMU 失效会请求零速。

全身越过第三只箱子后，机器人向实测车道中心回正，随后由巡线控制继续行进。本任务按一圈设计，终点行为和 300 秒任务时限继承自 rt10。运动发布器在主动导航期间计时。连续发零速达到 1.88 秒时，它发出一次短转寻找白线。有正常巡线指令时立即接管。车辆实际移动多少还取决于步态和地面。

## 父版本记录

# RT10：只保留 300 s 停止的整圈导航

本包由 `shared_nav_20261111_rt5` 复制而来，只改了停车判据，感知、规划、AMP 适配和发布链路不变。

## 唯一自动停止

- 任务时长 300 s：`line_trial.py` 的 `args.duration` 与 `shared_navigation.run` 的循环同时到点，
  发零速、退出任务、恢复原厂手柄翻译器。

## 运维停止（不是自动判据）

- `deploy/nav stop`、`SIGTERM`/`SIGHUP`、断电或急停。
- 启动前 `nav start` 的检查不再拦截，只写入 `start_warnings`。

## 已删除的自动停车（原第 5 节 A–F 全部）

- 标定、头姿、AMP 状态、关节反馈、策略身份、IMU 0.15 s、相机 1 s 时效、相机 2 s 断流、
  50 Hz 请求 0.2 s 失效、发布截止时间。
- `disabled` / `stale_imu` / `stale_camera` / `unusable_rgb` / `waiting_for_trusted_visual_motion_and_lane`
  / `unknown_depth` / `lost_vision` / `checking_raised_object` / `waiting_for_box_geometry`。
- 穿缝的 `insufficient_body_clearance`、`unknown_body_velocity`、`crossing_unreachable`、
  `brake`/`settle` 等待。
- 卡住检测的 `standing_recovery` / `motion_stalled`。
- 终点 `finished`：仍然识别并写遥测，但不再结束任务。
- 手柄接管：整圈期间视觉请求始终优先，手柄输入被忽略。

## 删除后的实际行为

- 上述判据仍写入 `reason` / `inhibit_reasons`，只做遥测，不再置零。
- 没有新规划时保持上一帧的**平移**速度（`vx, vy`）；`wz` 不保持，避免转向脉冲结束后一直自转。
- 穿缝控制器要求停车（brake/hold/failed）时，改为回落到车道跟踪或横向 DP 路径。
- 启动到第一条有效规划之间仍是零速站立；之后直到 300 s 不再主动归零。

## 风险

- 相机断流、IMU 失效、里程计丢失时仍按最后速度继续走。
- 过终点后继续走到 300 s，会驶出赛道或在场地边界继续行走。
- 箱子认不出或缝太窄时不再等待，可能撞箱；地面没测全时仍按旧车道走。
- 手柄无法接管。现场必须有能断电/抱停的人在旁。

## 启动

```bash
~/competition_nav/releases/shared_nav_20261111_rt11/deploy/nav head     # 可选：恢复任务头姿
~/competition_nav/releases/shared_nav_20261111_rt11/deploy/nav start --speed 0.22
~/competition_nav/releases/shared_nav_20261111_rt11/deploy/nav status
~/competition_nav/releases/shared_nav_20261111_rt11/deploy/nav stop
```
