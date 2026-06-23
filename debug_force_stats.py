"""检查 compute_norm_stats 数据管线中 state 是否包含 force 维度"""
import sys, numpy as np
sys.path.insert(0, 'src')
from openpi.training import config, data_loader

cfg = config.get_config('pi05_force_realman_finetune')
dc = cfg.data.create(cfg.assets_dirs, cfg.model)

# 与 compute_norm_stats.py 完全一致的 transforms
ds = data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)

class RmStr:
    def __call__(self, x):
        return {k:v for k,v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

ds = data_loader.TransformedDataset(
    ds,
    [*dc.repack_transforms.inputs, *dc.data_transforms.inputs, RmStr()]
)

# 模拟 RunningStats 计算
from openpi.shared.normalize import RunningStats
stats = RunningStats()
batch_size = 64
for bi in range(20):
    samples = [ds[bi * batch_size + j] for j in range(batch_size)]
    batch = {}
    for k in samples[0].keys():
        try:
            batch[k] = np.array([s[k] for s in samples])
        except Exception:
            batch[k] = [s[k] for s in samples]

    arr = batch['state']
    print(f'batch {bi}: shape={arr.shape}, force[0]={arr[0, 7:13]}')
    stats.update(arr)

result = stats.get_statistics()
print(f'\nmean: {result.mean}')
print(f'std:  {result.std}')
print(f'null dims: {[i for i,v in enumerate(result.mean) if v is None]}')
