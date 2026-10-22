# Mixed76 六模型推理交接包

整理日期：2026-10-20。本包用于交接 FastWAM、OpenWAM Alpha、Joint、Tri、Coarse-to-fine 和 Pi0.5 的推理接口及 G2/Piper 客户端。

先读 [中文交接文档](docs/六模型推理交接_分享版.md)，实现 server 或接入新语言时读 [中文通信协议](protocol/通信协议_TCP_v4.md)。

```text
docs/                           完整便携交接、依赖、启动命令、版本差异
clients/g2/                     G2客户端覆盖层、6份脱敏端点模板和协议测试
clients/piper/                  Piper客户端覆盖层、6份脱敏端点模板和IK/协议测试
clients/fr3_wuji/               FR3+Wuji EgoAV78 TCP v4观测/IK/fast真机适配层
protocol/                       中文协议、原始合同和可运行wire/mapping实现
tools/server_launch_examples.sh  六模型server通用启动模板（部署参数需填写）
tools/run_offline_tests.sh       本机回环测试，不访问机器人和模型服务器
tools/verify_manifest.py         文件数量/大小/SHA256独立校验
MANIFEST.json                   包内文件清单与SHA256
VALIDATION.md                   本次验证结果和验收边界
```

包里没有权重、训练数据、现场观测、真实端点、SSH密钥、虚拟环境或完整机器人驱动项目，也不包含完整六模型server实现。Server需要另外取得相应模型仓库、训练checkpoint及其配置/归一化/缓存依赖；本包提供启动接口与协议。

两份client是已有机器人项目的覆盖层：只做记录观测或回环协议测试可独立运行；真正执行需要完整G2项目或兼容EgoAV项目。解压不会启动任何进程或机器人。

安装最低无硬件测试依赖并验证：

```bash
python3 -m venv ../mixed76-test-env
../mixed76-test-env/bin/python -m pip install -r requirements-offline.txt
../mixed76-test-env/bin/python tools/verify_manifest.py
PYTHON=../mixed76-test-env/bin/python bash tools/run_offline_tests.sh
```

`requirements-offline.txt` 指“不访问硬件/真实模型”的测试环境；首次pip安装仍需要网络或自备wheel。它不是GPU模型部署或机器人硬件依赖锁文件。

收到包后需要自行配置：模型server代码、checkpoint目录、匹配Python运行时、可达的SSH别名和端点、机器人驱动/映射、相机与CAN配置。先用真实记录观测验证对应任务，再切到执行模式。

FR3+Wuji需另读 `clients/fr3_wuji/README.md`。该入口针对 `egoav78_chunk_v1`
78D/20Hz/32步服务，复用现有 EEF replay 的 cam0↔base、`fr3_link8` FK/IK 和
Pi05 观测链。真机入口与只读入口分离；真机入口要求显式受监督执行，并保留
Pi05 的所有权锁、关节限位、跟踪、反馈时效、20ms命令寿命和停止保护。
