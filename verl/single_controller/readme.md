## . 工作流程原理
### 资源池与 worker 创建

通过 RayResourcePool 和 RayClassWithInitArgs，结合 Ray 的 actor 机制，批量创建分布式 worker，并分配到不同节点和 GPU。

### 分布式 rank 信息同步

每个 worker 内部会根据 Megatron 并行库（如 megatron-core）获取自己的 rank 信息（tp_rank, dp_rank, pp_rank），并通过 Ray 的远程调用接口同步到 WorkerGroup。

### 全局信息收集

通过 rank_zero worker（通常是主节点），收集整个分布式训练的全局并行参数（如 tp_size, dp_size, pp_size），用于后续调度和数据分发。

### 数据分发与调度

WorkerGroup 内部会根据收集到的 rank 信息和全局信息，进行分布式数据分发和任务调度，保证 Megatron 并行训练的正确性和高效性。

### Ray 异步/同步远程调用

所有 worker 的方法调用（如 get_megatron_rank_info、get_megatron_global_info、init_megatron）都是通过 Ray 的远程 actor机制实现，可以异步或同步执行，保证分布式环境下的高效通信和状态同步。