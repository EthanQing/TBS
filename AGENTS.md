事实优先级：
1. 当前代码
2. migration / 配置
3. 测试
4. README
5. 文档

修改原则：
- 修改前先定位 domain owner
- 禁止为了修 bug 增加无期限 fallback
- 后处理修复前必须检查错误状态的产生源
- DB / filesystem / worker 副作用要明确 owner
- schema 变更必须 Alembic
- Docker 相关修改必须验证镜像

知识库：
当前处于重建阶段。
只有经过代码验证的稳定事实才允许加入。