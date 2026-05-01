"""Experiment logging and tracking utilities.

Provides unified logging to console, file, and optionally Weights & Biases.
Tracks training metrics, evaluation results, and experiment metadata.

Usage:
    from src.utils.logging import setup_logger, ExperimentTracker
    logger = setup_logger('experiment_name', log_dir='results/logs')
    tracker = ExperimentTracker(use_wandb=False)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def setup_logger(
    name: str,
    log_dir: str = "results/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and file handlers.

    Args:
        name: Logger name (used for log filename).
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    exp_logger = logging.getLogger(name)
    exp_logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if exp_logger.handlers:
        return exp_logger

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    exp_logger.addHandler(console)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log", encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    exp_logger.addHandler(file_handler)

    return exp_logger


class ExperimentTracker:
    """Track experiment metrics to file and optionally W&B.

    Args:
        experiment_name: Name for the experiment run.
        results_dir: Directory for saving results.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
    """

    def __init__(
        self,
        experiment_name: str = "mcgl",
        results_dir: str = "results",
        use_wandb: bool = False,
        wandb_project: str = "mcgl",
    ) -> None:
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self._run = None
        self._history: list[dict[str, Any]] = []

        if use_wandb:
            try:
                import wandb
                self._run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    reinit=True,
                )
            except ImportError:
                logger.warning("wandb not installed, falling back to file-only logging")
                self.use_wandb = False
            except Exception as e:
                logger.warning(f"wandb init failed: {e}, falling back to file-only logging")
                self.use_wandb = False

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics dict.

        Args:
            metrics: Dict of metric_name -> value.
            step: Optional step number (epoch, task, etc.).
        """
        entry = {**metrics}
        if step is not None:
            entry["step"] = step
        entry["timestamp"] = datetime.now().isoformat()
        self._history.append(entry)

        # Log to W&B
        if self.use_wandb and self._run is not None:
            import wandb
            wandb.log(metrics, step=step)

        # Log to console
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        step_str = f" [step {step}]" if step is not None else ""
        logger.info(f"Metrics{step_str}: {', '.join(parts)}")

    def log_config(self, config: dict) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dict.
        """
        if self.use_wandb and self._run is not None:
            import wandb
            wandb.config.update(config)

        config_path = self.results_dir / f"{self.experiment_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Config saved to {config_path}")

    def save_results(self, results: dict, filename: str) -> str:
        """Save results dict to JSON file.

        Args:
            results: Results dictionary.
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        out_path = self.results_dir / filename
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {out_path}")
        return str(out_path)

    def save_history(self) -> str:
        """Save full metric history to JSON.

        Returns:
            Path to saved history file.
        """
        path = self.results_dir / f"{self.experiment_name}_history.json"
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2, default=str)
        return str(path)

    def finish(self) -> None:
        """Finalize tracking (close W&B run, save history)."""
        self.save_history()
        if self.use_wandb and self._run is not None:
            import wandb
            wandb.finish()
