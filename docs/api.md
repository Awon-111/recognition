# 本地 API 与设备接入

默认地址为 [127.0.0.1:18765](http://127.0.0.1:18765)。OpenAPI 定义位于 [openapi.json](http://127.0.0.1:18765/openapi.json)。JSON 请求使用 UTF-8，坐标单位为 m，角度为度，时间为 s。

| 方法和地址 | 功能 |
|---|---|
| `GET /api/bootstrap` | 默认配置、工况、策略、活动模型。 |
| `GET /api/configs`、`GET /api/configs/{revision}` | 已保存版本及配置。 |
| `POST /api/configs` | 校验完整配置并保存内容哈希版本。请求结构见默认配置 JSON。 |
| `POST /api/runs` | 创建会话，传入 `case_id`、`strategy`、可选 `seed` 和 `revision`。 |
| `POST /api/runs/{id}/step` | 复核并执行一段或记录一次保持。传入当前 `last_index` 作为 `expected_index`。过期序号返回 409。 |
| `POST /api/runs/{id}/fault` | 接收 `{"fault":"offline"}`、`stale`、`steam`、`low` 或 `null`。清除后获取新帧。 |
| `POST /api/runs/{id}/disturb` | 施加模拟剪切扰动并重新观测。 |
| `GET /api/runs`、`GET /api/runs/{id}/trace`、`GET /api/runs/{id}/report` | 归档索引、带相机帧引用的完整过程、真值评测。 |
| `POST /api/batch` | 接收 `{"strategies":["balanced","gentle"],"revision":null}`。返回任务 ID。 |
| `GET /api/jobs/{id}`、`GET /api/batch/latest` | 训练/批量任务进度和最近评测结果。 |
| `GET /api/data` | 录像、标注检查、模型版本及验证结果。 |
| `POST /api/data/videos`、`POST /api/data/labels` | multipart 上传 `file`，标注额外传入 `video_id` 和可选 `furnace_id`。 |
| `POST /api/data/train` | 接收 `validation_ids` 和 `seed`，返回训练任务 ID。验证完成后替换活动模型。 |
| `GET /api/data/videos/{id}/stream` | NDJSON 逐帧回放。用 `clip_start_s` 与 `end_s` 指定片段，用 `start_s` 定位。启用 `analyze=true` 后输出增强与分段规划结果，支持 `strategy`、`revision`、`version` 和 `limit`。 |
| `POST /api/calibration/solve` | 传入录像 ID、倾角、同序 `pixel` 与 `plane_m` 对应点，返回单应矩阵与拟合 RMS。 |
| `GET /api/field/status`、`POST /api/field/status` | 接收和核查现场状态样本。状态包括 Unix 秒时间戳 `timestamp_s`、配置版本、倾角及四轴值。 |

录像流的 `observation` 含同帧图像的 YOLO 语义掩码、轮廓、帧号、采集时间和质量原因。`meta.model_name` 和 `meta.model_version` 标识当前模型。原始标签 `slag`、`iron`、`tool` 分别代表残渣、铁水和机械臂。`surface` 是液面投影的灰度索引网格，`data` 为 base64 字节。0 表示裸露铁水，1–250 为残渣色阶，254 为域外，255 为无法判别。仿真 `kind=concentration`，录像 `kind=appearance`。

`meta.reconstruction` 随帧给出 `basis`、参考倾角、水平液面参考几何与可用的工具 XY。未标定时 `basis=relative`，通过可见轮廓作相对投影。录像的实际倾角与工具高度需要同步设备数据。仅有效平面标定会在识别对象上提供米制位置和面积。录像回放期间执行器保持。

连续播放遵循源时间轴。处理积压时优先更新当前帧，`skipped` 返回跳过的迟到帧数。带 `limit` 的取帧请求按顺序读帧，用于准确定位与离线抽查。

分析模式按片段起点每秒更新一次规划。片段边界采用左闭右开区间，最后一次规划受 `end_s` 限制。单帧定位可用最多一秒重新规划，同时受片段结束时间限制。`meta.timing_ms` 记录预处理、YOLO 和投影耗时。`meta.plan` 在规划帧返回候选、选中轨迹、录像时间轴上的完成时间、截止时间、规划耗时和从读取目标帧开始的总耗时。超过截止时间时 `selected_id` 为空。相对投影用于参考设备仿真，现场需以标定和设备反馈建立动作截止时间及初始轴状态。

## 设备轨迹与校准入口

轨迹节点包含 `t_s`、相位、`joints`、`head_m`、倾角、水平液面和区段峰值轴速。相位从独立的 `tilt` 开始，后续扒渣段包含 `approach`、`skim`、`sweep`、`discharge`、`retreat`。相邻轴位置按 `q0 + (q1 - q0) * (3u² - 2u³)` 插值，`u` 为区段归一化时间。四轴键为 `extension_m`、`lift_m`、`swing_deg`、`pitch_deg`。

Python 侧 `slag.engine.Controller` 定义 `read_state`、`submit` 和 `hold`。现有实现绑定仿真设备。现场设备需实现适配器、控制握手、急停联锁和状态回读。现场状态 HTTP 入口做帧龄、行程、倾角和顺序校验，供联调准备。

平面标定使用原始图像像素与同一倾角下的液面坐标。求解结果先填入草稿，保存后的版本才用于录像投影。现场需测量安装位姿、畸变、设备到相机变换，并用独立点检验坐标误差。状态保持「现场待验证」，直到企业实炉完成测量。 [OpenCV 标定与单应矩阵文档](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

现场规划器可以替换为基于 CAD 与真实轴模型的碰撞检查、时间参数化和在线重规划接口。 [MoveIt 运动规划文档](https://moveit.picknik.ai/main/doc/concepts/motion_planning.html) · [MoveIt 混合规划文档](https://moveit.picknik.ai/main/doc/concepts/hybrid_planning/hybrid_planning.html)
