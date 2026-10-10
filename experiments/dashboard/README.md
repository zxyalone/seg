# 实验实时看板

纯 Python 标准库服务，HTML/CSS/JavaScript 不依赖外部 CDN。只读取调度状态、实验日志与验证结果，不导入训练库，不启动或控制 GPU 任务。

```bash
cd /root/autodl-tmp/Downstream_2
python experiments/dashboard/server.py --port 6006
```

网页每 5 秒刷新，调度器状态每 30 秒更新，GPU 每 15 秒采样。浏览器全部时间使用北京时间。AutoDL 的 6006「自定义服务」入口可访问；也可通过 SSH 转发 `ssh -L 6006:127.0.0.1:6006 <服务器连接参数>`，打开 `http://localhost:6006`。

- 阶段 0–3 显示另一台服务器导入的历史证据，保留冒烟显存峰值未采集说明。
- 阶段 4 / 5 / 6 来自实时 `_orchestration/state.json`、launcher 日志及 `eval_val/summary.json`。
- 当前轮次和已完成轮次分开；训练结束后等待验证，只有调度器确认完成才记为完整 run。
- 阶段 6 训练队列与 E10–E14 分析分开展示；分析待完成不会误报整个阶段已完成。独立分析完成后应补充其状态来源。
- 阶段 7 自动读取冻结标记与每个 run 的 `test_pred`、`eval_test`、`eval_external`，显示测试运行与评测数量；协议及结果完整性核验完成前不自动宣称整阶段完成。
- ETA 根据近期每轮耗时和共享 FIFO 队列估算，仅包含训练、验证、成本测量；每 run 预留 15 分钟验证，A8 参考 A6，实际会有波动。
- `/api/status` 返回看板数据，`/api/log?run=scheduler` 返回最近 100 行日志。只接受已知实验名称及核查日志，不开放任意文件读取。

常驻服务放在独立 tmux 会话 `phaser-dashboard`，与训练调度会话分开。访问日志仅记录非成功请求。服务输出位于结果盘 `_orchestration/dashboard.log`。
