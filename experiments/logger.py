"""
ExperimentLogger: Structured logging for reproducible experiments.
Captures metrics, hyperparameters, timings, and artifacts.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch
import numpy as np


class ExperimentLogger:
    """
    Unified logger for STREAM-FraudX experiments.

    Features:
    - Per-epoch metrics tracking
    - Hyperparameter recording
    - Wall-clock timing
    - Artifact path management
    - JSON + CSV output
    - Resume support
    """

    def __init__(self, run_id: str, output_dir: str = "artifacts/runs", resume: bool = False):
        """
        Initialize experiment logger.

        Args:
            run_id: Unique identifier for this run
            output_dir: Base directory for storing run artifacts
            resume: Whether to resume from existing run
        """
        self.run_id = run_id
        self.run_dir = Path(output_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self.metadata = {
            'run_id': run_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'status': 'running'
        }

        # Initialize metric storage
        self.metrics = []
        self.hyperparameters = {}
        self.artifacts = {}

        # Load existing data if resuming
        if resume and (self.run_dir / "metrics.json").exists():
            self._load_existing()

        # Initialize timing
        self.start_wall_time = time.time()
        self.epoch_start_time = None

    def _load_existing(self):
        """Load existing run data for resuming."""
        with open(self.run_dir / "metrics.json", 'r') as f:
            data = json.load(f)
            self.metadata.update(data.get('metadata', {}))
            self.metrics = data.get('metrics', [])
            self.hyperparameters = data.get('hyperparameters', {})
            self.artifacts = data.get('artifacts', {})

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters for this run.

        Args:
            params: Dictionary of hyperparameters
        """
        self.hyperparameters.update(params)
        self._save_metadata()

    def log_seeds(self, seed: int, additional_info: Optional[Dict] = None):
        """
        Log random seeds used for reproducibility.

        Args:
            seed: Main random seed
            additional_info: Additional seed information
        """
        seed_info = {
            'seed': seed,
            'pytorch_seed': torch.initial_seed(),
            'numpy_seed': seed,
        }
        if additional_info:
            seed_info.update(additional_info)

        self.metadata['seeds'] = seed_info
        self._save_metadata()

    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def end_epoch(self, metrics: Dict[str, float]):
        """
        Log metrics at the end of an epoch.

        Args:
            metrics: Dictionary of metric name -> value
        """
        if self.epoch_start_time is None:
            raise ValueError("Must call start_epoch before end_epoch")

        epoch_duration = time.time() - self.epoch_start_time

        epoch_record = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': epoch_duration,
            'metrics': metrics
        }

        self.metrics.append(epoch_record)
        self._save_metrics()

        # Reset epoch timer
        self.epoch_start_time = None

    def log_artifact(self, name: str, path: str, artifact_type: str = "file"):
        """
        Register an artifact path.

        Args:
            name: Artifact name
            path: Path to artifact
            artifact_type: Type of artifact (file, checkpoint, plot, etc.)
        """
        self.artifacts[name] = {
            'path': str(path),
            'type': artifact_type,
            'timestamp': datetime.now().isoformat()
        }
        self._save_metadata()

    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model architecture information.

        Args:
            model_info: Dictionary with model details (params, architecture, etc.)
        """
        self.metadata['model_info'] = model_info
        self._save_metadata()

    def finalize(self, status: str = "completed", final_metrics: Optional[Dict] = None):
        """
        Finalize the experiment run.

        Args:
            status: Final status (completed, failed, interrupted)
            final_metrics: Optional final test metrics
        """
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = time.time() - self.start_wall_time
        self.metadata['status'] = status

        if final_metrics:
            self.metadata['final_metrics'] = final_metrics

        self._save_metadata()
        self._save_metrics()

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'hyperparameters': self.hyperparameters,
                'artifacts': self.artifacts
            }, f, indent=2)

    def _save_metrics(self):
        """Save metrics to JSON and CSV."""
        # Save JSON
        metrics_json = self.run_dir / "metrics.json"
        with open(metrics_json, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'hyperparameters': self.hyperparameters,
                'artifacts': self.artifacts,
                'metrics': self.metrics
            }, f, indent=2)

        # Save CSV (flattened metrics)
        metrics_csv = self.run_dir / "metrics.csv"
        if self.metrics:
            with open(metrics_csv, 'w', newline='') as f:
                # Get all metric keys
                metric_keys = set()
                for record in self.metrics:
                    metric_keys.update(record['metrics'].keys())
                metric_keys = sorted(metric_keys)

                fieldnames = ['epoch', 'timestamp', 'duration_seconds'] + metric_keys
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for record in self.metrics:
                    row = {
                        'epoch': record['epoch'],
                        'timestamp': record['timestamp'],
                        'duration_seconds': record['duration_seconds']
                    }
                    row.update(record['metrics'])
                    writer.writerow(row)

    def get_best_metric(self, metric_name: str, mode: str = "max") -> Optional[Dict]:
        """
        Get the best value for a specific metric.

        Args:
            metric_name: Name of the metric
            mode: 'max' or 'min'

        Returns:
            Dictionary with epoch and value, or None if not found
        """
        if not self.metrics:
            return None

        values = []
        for record in self.metrics:
            if metric_name in record['metrics']:
                values.append({
                    'epoch': record['epoch'],
                    'value': record['metrics'][metric_name]
                })

        if not values:
            return None

        if mode == "max":
            best = max(values, key=lambda x: x['value'])
        else:
            best = min(values, key=lambda x: x['value'])

        return best

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment."""
        summary = {
            'run_id': self.run_id,
            'status': self.metadata.get('status'),
            'duration': self.metadata.get('duration_seconds'),
            'num_epochs': len(self.metrics),
            'hyperparameters': self.hyperparameters,
        }

        # Add best metrics if available
        if self.metrics and self.metrics[-1].get('metrics'):
            summary['final_metrics'] = self.metrics[-1]['metrics']

        return summary

    @staticmethod
    def list_runs(output_dir: str = "artifacts/runs") -> List[str]:
        """
        List all experiment runs.

        Args:
            output_dir: Base directory for runs

        Returns:
            List of run IDs
        """
        runs_path = Path(output_dir)
        if not runs_path.exists():
            return []

        return [d.name for d in runs_path.iterdir() if d.is_dir()]

    @staticmethod
    def load_run(run_id: str, output_dir: str = "artifacts/runs") -> 'ExperimentLogger':
        """
        Load an existing run.

        Args:
            run_id: Run identifier
            output_dir: Base directory for runs

        Returns:
            ExperimentLogger instance
        """
        logger = ExperimentLogger(run_id, output_dir, resume=True)
        return logger
