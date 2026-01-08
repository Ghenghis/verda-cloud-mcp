"""
Distributed Training - Multi-node/multi-GPU setup helpers.
MEGA-TOOL bundling 10 functions into 1 tool.
"""


def distributed_training(action: str = "guide", nodes: int = 2, gpus_per_node: int = 8, **kwargs) -> str:
    """
    MEGA-TOOL: Distributed Training (10 functions).

    Actions: guide, deepspeed, fsdp, accelerate, torchrun, nccl,
    config_multi_node, config_multi_gpu, benchmark, troubleshoot
    """
    if action == "guide":
        return """
🌐 DISTRIBUTED TRAINING GUIDE
═══════════════════════════════════════════════════

📊 SCALING EFFICIENCY
GPUs │ Speedup │ Efficiency
─────┼─────────┼───────────
  1  │  1.0x   │   100%
  2  │  1.85x  │   92.5%
  4  │  3.5x   │   87.5%
  8  │  6.5x   │   81.25%
 16  │  12x    │   75%
 32  │  22x    │   68.75%

🛠️ RECOMMENDED TOOLS
1. HuggingFace Accelerate (easiest)
   distributed_training(action='accelerate')

2. PyTorch FSDP (native, efficient)
   distributed_training(action='fsdp')

3. DeepSpeed ZeRO-3 (most features)
   distributed_training(action='deepspeed')

💡 TIP: Start with Accelerate, upgrade to DeepSpeed for 70B+ models
"""

    elif action == "deepspeed":
        return f'''# DeepSpeed ZeRO-3 Configuration
# Nodes: {nodes} | GPUs/Node: {gpus_per_node} | Total: {nodes * gpus_per_node}

# ds_config.json
{{
  "train_batch_size": {nodes * gpus_per_node * 4},
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "fp16": {{"enabled": true}},
  "zero_optimization": {{
    "stage": 3,
    "offload_optimizer": {{"device": "cpu", "pin_memory": true}},
    "offload_param": {{"device": "cpu", "pin_memory": true}},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }},
  "activation_checkpointing": {{
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }}
}}

# Launch command
deepspeed --num_gpus={gpus_per_node} --num_nodes={nodes} \\
  --hostfile=hostfile.txt \\
  --master_addr=node0 --master_port=29500 \\
  train.py --deepspeed ds_config.json
'''

    elif action == "fsdp":
        return f'''# PyTorch FSDP (Fully Sharded Data Parallel)
# Best for: 7B-70B models on multi-GPU

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# FSDP configuration
fsdp_config = dict(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    cpu_offload=CPUOffload(offload_params=True),
    auto_wrap_policy=transformer_auto_wrap_policy,
)

model = FSDP(model, **fsdp_config)

# Launch: torchrun --nproc_per_node={gpus_per_node} train.py
'''

    elif action == "accelerate":
        return f'''# HuggingFace Accelerate (EASIEST!)
# Automatically handles: DDP, FSDP, DeepSpeed, Multi-node

# 1. Install
pip install accelerate

# 2. Configure (interactive)
accelerate config

# 3. Your training script (minimal changes!)
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    loss = model(**batch).loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

# 4. Launch
accelerate launch --num_processes={nodes * gpus_per_node} train.py

# Multi-node launch
accelerate launch --num_machines={nodes} --num_processes={gpus_per_node} \\
  --machine_rank=$RANK --main_process_ip=node0 train.py
'''

    elif action == "torchrun":
        return f'''# torchrun (PyTorch native launcher)

# Single node, multi-GPU
torchrun --nproc_per_node={gpus_per_node} train.py

# Multi-node
# On node 0 (master):
torchrun --nproc_per_node={gpus_per_node} --nnodes={nodes} \\
  --node_rank=0 --master_addr=node0 --master_port=29500 train.py

# On node 1:
torchrun --nproc_per_node={gpus_per_node} --nnodes={nodes} \\
  --node_rank=1 --master_addr=node0 --master_port=29500 train.py
'''

    elif action == "nccl":
        return '''# NCCL Environment Variables (for debugging/tuning)

# Basic debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Performance tuning
export NCCL_IB_DISABLE=0          # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=2       # GPU Direct RDMA
export NCCL_P2P_LEVEL=NVL         # NVLink P2P

# Timeout (increase for large models)
export NCCL_TIMEOUT=1800          # 30 minutes

# For cloud/firewall issues
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
'''

    elif action == "config_multi_node":
        return f'''# Multi-Node Configuration

# hostfile.txt (for DeepSpeed)
node0 slots={gpus_per_node}
node1 slots={gpus_per_node}

# SSH setup (passwordless)
ssh-keygen -t rsa
ssh-copy-id node1

# Shared filesystem required!
# Mount /workspace on all nodes via NFS or shared storage

# Network requirements
# - High bandwidth between nodes (25Gbps+)
# - Low latency (<1ms)
# - Open ports: 29500 (master), NCCL ports
'''

    elif action == "config_multi_gpu":
        return f'''# Multi-GPU (Single Node) Configuration
# Simpler than multi-node - just set devices!

# Option 1: Environment variable
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Option 2: In code
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Option 3: torchrun
torchrun --nproc_per_node={gpus_per_node} train.py

# Verify
python -c "import torch; print(torch.cuda.device_count())"
'''

    elif action == "benchmark":
        return f"""
📊 DISTRIBUTED SCALING BENCHMARK
═══════════════════════════════════════════════════

Configuration: {nodes} nodes × {gpus_per_node} GPUs = {nodes * gpus_per_node} total

Expected Performance:
┌──────────┬──────────┬────────────┬──────────────┐
│ GPUs     │ Speedup  │ Efficiency │ Throughput   │
├──────────┼──────────┼────────────┼──────────────┤
│ 1        │ 1.0x     │ 100%       │ 100 tok/s    │
│ {gpus_per_node}        │ {gpus_per_node * 0.85:.1f}x     │ {85}%       │ {gpus_per_node * 85} tok/s    │
│ {nodes * gpus_per_node}       │ {nodes * gpus_per_node * 0.75:.1f}x    │ {75}%       │ {nodes * gpus_per_node * 75} tok/s   │
└──────────┴──────────┴────────────┴──────────────┘

💡 NVLink GPUs (H100, B300) scale better than PCIe
"""

    elif action == "troubleshoot":
        return """
🔧 DISTRIBUTED TRAINING TROUBLESHOOTING
═══════════════════════════════════════════════════

❌ NCCL Timeout
   → Increase: export NCCL_TIMEOUT=1800
   → Check network connectivity between nodes
   → Verify firewall allows NCCL ports

❌ OOM (Out of Memory)
   → Reduce batch size
   → Enable gradient checkpointing
   → Use DeepSpeed ZeRO-3 offloading
   → Use FSDP with CPU offload

❌ Slow Training
   → Check GPU utilization (nvidia-smi)
   → Verify NVLink/InfiniBand is being used
   → Increase batch size if GPU util < 90%
   → Use gradient accumulation

❌ Process Hangs
   → All nodes must start simultaneously
   → Check master_addr is correct
   → Verify node_rank is different per node

❌ Loss NaN/Inf
   → Lower learning rate
   → Add gradient clipping
   → Check data for invalid values
   → Use mixed precision carefully
"""

    return "Actions: guide, deepspeed, fsdp, accelerate, torchrun, nccl, config_multi_node, config_multi_gpu, benchmark, troubleshoot"
