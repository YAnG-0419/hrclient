# Mixed76 TCP v4 通信协议

协议版本4，表示名称固定 `head12_wrist18_hand46_rot6d`。本文件按包内实际client/wire代码整理。完整原始合同保存在 `PROTOCOL_V4_original.md`，Python参考实现为本目录 `_ours_protocol.py` 和 `mixed76_contract.py`。

## 1. 传输层

使用普通TCP流，可经SSH端口转发。不是HTTP，也不是WebSocket。不需要URL路径。

每条消息：`4字节无符号大端长度 N`＋`N字节 UTF-8 JSON`。N只计JSON体，必须在1到16MiB之间。TCP的recv可能只返回部分字节，必须循环读满长度头和正文。连接关闭时未完成的消息作废。

JSON顶层必须是对象。`allow_nan=False`，不发送NaN/Inf。不得使用pickle。一个连接同步发送一条请求并等到响应，随后再发送下一条。

NumPy数组编码成如下JSON对象：

```json
{"__array__":true,"dtype":"<f4","shape":[76],"data":"BASE64_OF_CONTIGUOUS_RAW_BYTES"}
```

`data`是C连续排列的原始字节Base64，不是图像JPEG/PNG文件。float32通常`<f4`，uint8为`|u1`，float64通常`<f8`；decoder接受uint8/float32/float64，校验shape与字节数一致，维数不超过4，每个数组不超过16MiB。消息整体还需不超过16MiB。endian由dtype字符串表达。

`head_rgb`通过uint8数组包传输，是HWC的RGB原图；不是BGR，不在client做模型专属裁剪/归一化。当前Pi adapter要求480×640×3，实际跨六模型验收统一使用该尺寸。client自身允许的图像范围比某些server宽，不能把client本地检查通过当成各server已接受。

`action_mask`是普通JSON布尔列表，因为数组decoder不接受bool dtype。`actions`允许使用上述float32数组包；现有client的validate_actions也接受可转换为float32的二维JSON数值列表。

## 2. 会话流程

```text
打开TCP连接
  -> info：核对身份和接口
  -> reset：取得该连接的新session_id
  -> infer：输入一份新观测，返回一个完整32帧chunk
  -> infer：下次观测的新请求
  -> 可选reset：旧session失效
关闭连接；下次重连重新info/reset
```

session归属于TCP连接，不能跨连接复用。request_id需唯一，不得重放旧请求。重连后重新观测，不把旧动作自动重发给机器人。reset清理时序状态和相关缓存，不卸载权重，也不要求删除不可变编译内核。

## 3. info

请求：

```json
{"cmd":"info"}
```

响应字段：

| 字段 | 值/语义 |
|---|---|
| `protocol_version` | 整数4 |
| `model` | fastwam/alpha/joint/tri/coarse_fine/pi05 |
| `representation` | head12_wrist18_hand46_rot6d |
| `state_dim`, `action_dim` | 均76 |
| `state_units`, `action_units` | physical |
| `action_mode` | absolute |
| `horizon`, `dt_s` | 32、0.05 |
| `camera_keys` | ["head_rgb"] |
| `embodiments` | 声明支持的3b/g2 |
| `checkpoint`, `checkpoint_step` | 权重身份和训练步数 |
| `normalization_sha256` | 所用归一化资产/合同SHA256；不同adapter选择的具体文件可能不同，按同部署核验 |
| `deployment_id` | 绑定权重、配置、适配代码和采样设置的部署身份 |
| `prompts` | task到精确训练指令的映射 |
| `image_preprocess` | 模型自己的预处理说明 |
| `denoise_steps`, `inference_compile` | 实际推理设置 |
| `weights_strict` | true |
| `coordinate_frames`, `component_units` | 坐标与单位来源及验证状态 |

当前client自动检查固定接口字段、checkpoint/normalization/deployment身份非空，以及可选的精确checkpoint身份。它没有自动认证坐标系和单位来源，也不会自动把info所有描述字段与外部配置逐项比较。操作方仍须核对这些字段。

## 4. reset

请求：

```json
{"cmd":"reset","request_id":"reset-unique-001"}
```

响应示意：

```json
{"protocol_version":4,"request_id":"reset-unique-001","deployment_id":"DEPLOYMENT_ID","session_id":"NEW_OPAQUE_ID","ok":true}
```

client严格检查request_id、deployment_id、版本和ok，并保存非空session_id。未reset就infer、reset后使用旧session应返回STALE_SESSION。

## 5. infer

下述state/head_rgb用数组占位，真正发包时必须用wire数组编码：

```text
{
  "cmd": "infer",
  "request_id": "unique-request-001",
  "session_id": "NEW_OPAQUE_ID",
  "sample_ns": 123456789,
  "task": "book",
  "embodiment": "3b",
  "observation": {
    "state": float32[76],
    "head_rgb": uint8[480,640,3]
  }
}
```

`state`是未经归一化的训练约定物理值，所有元素有限，inactive槽位严格0。`task`为basket/book/diguan/yifu/dengpao，`embodiment`显式传3b/g2。server从prompts中选指令，client不传自由改写prompt。

`sample_ns`是client本机单调时钟整数，不是UTC时间。server原样回传，用于关联；不同机器的monotonic起点不可直接比较。观测年龄在client自身时钟计算。

响应：

```text
{
  "protocol_version": 4,
  "request_id": "unique-request-001",
  "session_id": "NEW_OPAQUE_ID",
  "sample_ns": 123456789,
  "deployment_id": "DEPLOYMENT_ID",
  "embodiment": "3b",
  "representation": "head12_wrist18_hand46_rot6d",
  "action_units": "physical",
  "dt_s": 0.05,
  "actions": float32[32,76],
  "action_mask": bool-list[76],
  "inference_seconds": 0.123
}
```

request/session/sample/deployment身份必须对应当前请求。actions有限、inactive列严格0、mask与本体精确一致，模型耗时有限且非负。client在返回动作前检查这些条件。

每次infer从当前观测计算整个新chunk。不能重复同一步32次来伪造chunk。client选择执行前N帧，然后重新采集；server不维护机器人执行队列。模型内部cache由各自原生实现管理，不改变本协议的请求身份约束。

## 6. 错误

```json
{"error":{"code":"STALE_SESSION","message":"Reset this connection before inference"},"request_id":"unique-request-001"}
```

常用code：`INVALID_REQUEST`、`MODEL_ERROR`、`STALE_SESSION`。失败响应不得附可执行actions。client抛出错误，不自动降级到其他模型，不重放上次chunk。

## 7. 76D动作布局与mask

所有slice为Python半开区间，index从0开始。

| 区间 | 内容 | Piper/3b有效 | G2有效 |
|---|---|---|---|
| 0:9 | 相机臂/头部EEF xyz＋rot6D | 是 | 否 |
| 9:12 | G2头部3维关节 | 否 | 是 |
| 12:21 | 左腕xyz＋rot6D | 是 | 是 |
| 21:30 | 右腕xyz＋rot6D | 是 | 是 |
| 30:52 | 左手22维 | 否 | 是 |
| 52 | 左夹爪 | 是 | 否 |
| 53:75 | 右手22维 | 否 | 是 |
| 75 | 右夹爪 | 是 | 否 |

Piper有效29维；G2有效65维。网络始终传76维，不能压缩成29/65维后省略槽位。Alpha内部80D不能泄露为公共wire的80D。

rot6D是旋转矩阵**前两列逐列拼接**，不是取前两行或随意reshape。原生四元数为xyzw。Piper原生23D，G2原生61D；转换应复用 `mixed76_contract.encode_state/decode_actions`，不要自行猜测维度顺序。

## 8. 最小Python调用

在G2客户端目录下：

```python
import sys
import numpy as np
sys.path.insert(0, "scripts")
from client_mixed76 import Mixed76Client
from mixed76_contract import encode_state

with np.load("/path/to/book_observation.npz", allow_pickle=False) as data:
    state = data["state"]
    rgb = data["head_rgb"]
if state.shape != (76,):
    state = encode_state(state, "3b")
client = Mixed76Client("127.0.0.1", 18849, model="fastwam")
try:
    actions = client.infer(state, rgb, task="book", embodiment="3b")
    print(actions.shape)
finally:
    client.close()
```

该示例假设已具备本地可达端点/隧道，不导入硬件、不执行动作。server真实模型预处理、输出、坐标和时延仍需单独验收。
