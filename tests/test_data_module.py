"""
Tests for data module.
"""

import pytest
import torch
from stream_fraudx.data.module import (
    LabelLatencyQueue,
    MicroBatchStream,
    StreamDataModule
)


def test_label_latency_queue():
    """Test label latency queue."""
    queue = LabelLatencyQueue(delay_batches=3)

    # Add batches
    for i in range(5):
        samples = {'continuous': torch.randn(10, 5)}
        labels = torch.randint(0, 2, (10,))
        queue.add_batch(samples, labels)

        ready = queue.get_ready_labels()

        if i < 3:
            assert ready is None, f"No labels should be ready at step {i}"
        else:
            assert ready is not None, f"Labels should be ready at step {i}"
            assert 'labels' in ready


def test_microbatch_stream():
    """Test micro-batch streaming."""
    from stream_fraudx.data.synthetic_data import SyntheticFraudDataset, collate_fn

    dataset = SyntheticFraudDataset(num_samples=1000, num_nodes=100)

    stream = MicroBatchStream(
        dataset=dataset,
        microbatch_size=50,
        window_seconds=30,
        collate_fn=collate_fn
    )

    # Test iteration
    batch_count = 0
    for batch in stream:
        batch_count += 1
        assert 'continuous' in batch
        assert 'labels' in batch
        assert 'timestamp' in batch
        if batch_count >= 5:  # Test first 5 batches
            break

    assert batch_count == 5


def test_stream_data_module_synthetic():
    """Test StreamDataModule with synthetic data."""
    data_module = StreamDataModule(
        dataset_name='synthetic',
        synthetic_params={
            'num_samples': 1000,
            'num_nodes': 100,
            'num_continuous': 5,
            'num_categorical': 3
        }
    )

    data_module.setup()

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert data_module.test_dataset is not None

    # Test dataloaders
    train_loader = data_module.train_dataloader(batch_size=32)
    batch = next(iter(train_loader))

    assert 'continuous' in batch
    assert 'categorical' in batch
    assert 'labels' in batch

    # Test streaming
    stream = data_module.stream_dataloader(split='test')
    microbatch = next(iter(stream))

    assert 'continuous' in microbatch
    assert 'timestamp' in microbatch
