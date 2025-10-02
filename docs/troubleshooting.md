# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. PyTorch CUDA Version Mismatch

**Error**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
Check CUDA version and reinstall PyTorch:
```bash
nvidia-smi  # Check CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA
```

#### 2. Missing Dependencies

**Error**:
```
ModuleNotFoundError: No module named 'lightgbm'
```

**Solution**:
```bash
pip install -r requirements.txt
```

### Data Issues

#### 3. Kaggle API Not Configured

**Error**:
```
OSError: Could not find kaggle.json
```

**Solution**:
1. Create Kaggle API token at https://www.kaggle.com/account
2. Place `kaggle.json` in `~/.kaggle/`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

#### 4. Dataset Download Fails

**Error**:
```
403 Forbidden: You must accept competition rules
```

**Solution**:
Visit dataset page and accept rules:
- IEEE-CIS: https://www.kaggle.com/c/ieee-fraud-detection/rules
- PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
- Elliptic: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

### Training Issues

#### 5. CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce batch size: `--batch_size 128`
- Reduce model dimensions:
  ```bash
  python main.py --tgt_node_dim 64 --ttt_d_model 64 --batch_size 128
  ```
- Use gradient accumulation:
  ```python
  # In trainer, accumulate over N steps
  loss = loss / accumulation_steps
  loss.backward()
  if (step + 1) % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
  ```

#### 6. Training Crashes with IRM Loss

**Error**:
```
RuntimeError: Trying to backward through the graph a second time
```

**Solution**:
This should be fixed in the updated trainer. The fix uses detach:
```python
embeddings_detached = {k: v.detach().requires_grad_(True)
                      for k, v in embeddings.items()}
irm_penalty = irm_loss(embeddings_detached['fused'], labels, time_slices)
```

If still occurring, disable IRM temporarily:
```bash
python main.py --irm_weight 0.0
```

#### 7. NaN Loss During Training

**Possible Causes**:
- Learning rate too high
- Numerical instability in loss

**Solutions**:
1. Reduce learning rate: `--learning_rate 1e-4`
2. Add gradient clipping (already implemented):
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
3. Check for invalid inputs:
   ```python
   assert not torch.isnan(features).any()
   assert not torch.isinf(features).any()
   ```

### Model Issues

#### 8. Pretrained Checkpoint Loading Fails

**Error**:
```
RuntimeError: Error(s) in loading state_dict
```

**Solution**:
This is expected when loading Stage A weights into Stage B (missing classifier head). Use `strict=False`:
```python
model.load_state_dict(checkpoint, strict=False)
```

#### 9. Adapters Not Updating

**Symptom**: Validation metrics don't improve with adapters

**Solutions**:
1. Check adapters are trainable:
   ```python
   for name, param in model.named_parameters():
       if 'adapter' in name:
           print(f"{name}: requires_grad={param.requires_grad}")
   ```
2. Increase adapter learning rate: `--learning_rate 2e-3`
3. Verify adapters are in optimizer:
   ```python
   adapter_params = [p for n, p in model.named_parameters()
                     if 'adapter' in n and p.requires_grad]
   print(f"Adapter params: {len(adapter_params)}")
   ```

### Performance Issues

#### 10. Slow Training

**Solutions**:
1. Enable CUDA: Ensure `torch.cuda.is_available() == True`
2. Increase batch size (if memory allows)
3. Reduce number of GNN layers: `--tgt_num_layers 2`
4. Use mixed precision:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       logits = model(batch)
       loss = criterion(logits, labels)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```
5. Disable validation during training: `--eval_every 10`

#### 11. High Memory Usage

**Solutions**:
- Reduce memory bank size:
  ```python
  model.tgt.memory_bank.max_size = 500  # Default: 1000
  ```
- Clear cache periodically:
  ```python
  if epoch % 5 == 0:
      torch.cuda.empty_cache()
  ```
- Use CPU offloading for large datasets

### Streaming Issues

#### 12. Active Learning Not Selecting Samples

**Symptom**: `labels_used_today == 0`

**Solutions**:
1. Check budget: `--label_budget 100`
2. Verify conformal predictor is calibrated:
   ```python
   print(f"Calibration size: {len(conformal.calibration_buffer)}")
   print(f"Quantile: {conformal.quantile}")
   ```
3. Lower uncertainty threshold

#### 13. Drift Not Detected

**Symptom**: No drift events logged

**Solutions**:
1. Lower drift threshold: `--drift_threshold 0.05`
2. Increase window size: `--drift_window 200`
3. Check drift detector is receiving labels:
   ```python
   print(f"Window size: {len(drift_detector.scores_window)}")
   ```

### Evaluation Issues

#### 14. Poor Test Metrics

**Possible Causes**:
- Overfitting
- Class imbalance not handled
- No pretraining

**Solutions**:
1. Add regularization: `--weight_decay 1e-4`
2. Use focal loss (already default)
3. Run Stage A first: `python pretrain.py`
4. Check for label leakage
5. Verify train/val/test split is temporal (not random)

#### 15. Metrics Not Saved

**Error**:
```
FileNotFoundError: outputs/results.json
```

**Solution**:
Ensure output directory exists:
```bash
mkdir -p outputs
python main.py --output_dir outputs
```

### Testing Issues

#### 16. Tests Failing

**Error**:
```
ImportError: cannot import name 'STREAMFraudX'
```

**Solution**:
Install package in development mode:
```bash
pip install -e .
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

#### 17. CUDA Tests Fail on CPU

**Error**:
```
RuntimeError: Expected all tensors to be on the same device
```

**Solution**:
Tests automatically detect device, but you can force CPU:
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/
```

## Debugging Tips

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Gradient Flow

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### Profile Code

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(batch)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Visualize Model

```python
from torchviz import make_dot
logits = model(batch)
make_dot(logits, params=dict(model.named_parameters())).render("model_graph", format="png")
```

## Getting Help

If issues persist:

1. Check GitHub Issues: Search for similar problems
2. Enable debug mode: `python main.py --debug`
3. Provide:
   - Error traceback
   - Environment: `pip list`, `nvidia-smi`
   - Command used
   - Minimal reproducible example

## Known Limitations

1. **Graph Construction**: Assumes bipartite user-merchant graph. For other topologies, modify `synthetic_data.py`
2. **Memory Bank**: Fixed size (1000). For larger graphs, consider distributed memory
3. **Streaming**: Simulated, not true online deployment. For production, integrate with stream processor (Kafka, Flink)
4. **Datasets**: Loaders assume specific formats. Adapt for custom datasets
5. **Multi-GPU**: Not implemented. Use `torch.nn.DataParallel` or `DistributedDataParallel`

## Performance Tuning Checklist

- [ ] CUDA enabled: `torch.cuda.is_available() == True`
- [ ] Batch size maximized (within memory limits)
- [ ] Mixed precision training enabled
- [ ] Gradient checkpointing for large models
- [ ] Efficient data loading: `num_workers > 0` in DataLoader
- [ ] Model compiled: `model = torch.compile(model)` (PyTorch 2.0+)
- [ ] Memory bank size appropriate for dataset
- [ ] Validation frequency not too high
- [ ] Logging/checkpointing not on hot path
