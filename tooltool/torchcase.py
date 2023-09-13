"""
Train-case is a combination of model, optimizer, scheduler, logger.
Only single-gpu case is supported.
There can be multiple train-cases trained together.
Auto-save and 'load the latest'

Example usage:

```
case = TrainCase(...)

if case.checkpoint_list:
    case.checkpoint_load()

for epoch in range(100):
    with TrainCase.run_epoch([case]):
        for batch_input, batch_output in data:
            batch_input, batch_output = case.to_device_all(batch_input, batch_output)
            train_loss = compute_loss(case.model(batch_input), batch_output)
            case.backprop(loss) # backward + optim.step + optim.zero_grad
            case.accumulate_until_collected('train_loss', train_loss)
            test_loss = ...
            case.accumulate_over_epoch('test_loss', test_loss)
```


"""
from __future__ import annotations

import contextlib
import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


def _compute_mean(sum_and_count: list[tuple]) -> float:
    total_sum = 0
    total_count = 0
    for s, c in sum_and_count:
        total_sum += s
        total_count += c

    return total_sum / total_count if total_count > 0 else np.nan


@dataclasses.dataclass
class TrainCase:
    # what makes this name different from anything else? Be short and precise
    name: str
    device: torch.device
    optimizer: torch.optim.Optimizer
    model: torch.nn.Module
    writer: SummaryWriter
    scheduler: torch.optim.lr_scheduler.LambdaLR = None
    step: int = 1
    epoch: int = 1
    auto_save_each_n_epochs: int = 20

    _epoch_accumulated_metrics = None

    def __post_init__(self):
        self._accumulated_metrics = {}
        self._cache = {}
        assert " " not in self.name
        assert self.writer.log_dir.endswith(self.name), (self.name, self.writer.log_dir)

    def backprop(self, loss):
        loss.backward()
        self.accumulate_until_collected("train_loss", loss, 64)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1

    def to_device(self, value: Union[torch.Tensor, np.ndarray, pd.Series]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(self.device, non_blocking=True)
        if isinstance(value, pd.Series):
            return torch.from_numpy(value.values).to(self.device, non_blocking=True)
        raise NotImplementedError(f"Unknown value: {value} {type(value)}")

    def to_device_all(self, *values) -> list[torch.Tensor]:
        return [self.to_device(x) for x in values]

    @staticmethod
    def as_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    def report_scalar(self, tag: str, value: Union[torch.Tensor, float]):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.step)

    def accumulate_until_collected(self, tag: str, value: torch.Tensor, max_collect=100):
        self._accumulated_metrics.setdefault(tag, []).append(value.detach().cpu().item())
        if len(self._accumulated_metrics[tag]) >= max_collect:
            self.report_scalar(tag, float(np.mean(self._accumulated_metrics[tag])))
            self._accumulated_metrics[tag] = []

    def accumulate_over_epoch(self, tag: str, value: torch.Tensor):
        self._epoch_accumulated_metrics.setdefault(tag, []).append(value.detach().cpu().item())

    def _checkpoint_folder(self) -> Path:
        return Path(self.writer.get_logdir()).joinpath("saved_models", self.name)

    def checkpoint_save(self) -> Path:
        path = self._checkpoint_folder().joinpath(f"{self.epoch:>05}.pth")
        path.parent.mkdir(exist_ok=True, parents=True)
        path = str(path.absolute())
        torch.jit.save(self.model, path)
        return Path(path)

    def checkpoint_load(self, epoch_glob="*"):
        """Loads the latest checkpoint by default.
        You can provide a glob regexp for narrowing search, e.g. epoch_glob='003*'"""
        paths = self._checkpoint_folder().glob(f"{epoch_glob}.pth")
        path = list(sorted(paths))[-1]
        print(f"loading from {path}")
        self.model = torch.jit.load(str(path.absolute())).to(self.device)

    def checkpoint_list(self) -> list[Path]:
        return list(self._checkpoint_folder().glob("*.pth"))

    def checkpoint_delete_all(self):
        paths = list(self._checkpoint_folder().glob("*.pth"))
        print(f"Deleted {len(paths)} files")
        for p in paths:
            p.unlink()

    @staticmethod
    @contextlib.contextmanager
    def run_epoch(cases: list[TrainCase]):
        collected_names = set()
        for case in cases:
            assert case.name not in collected_names, "duplicate names"
            collected_names.add(case.name)

            case._epoch_accumulated_metrics = defaultdict(list)

        yield

        for case in cases:
            case.epoch += 1
            case.report_scalar("epoch", case.epoch)
            for tag, values in case._epoch_accumulated_metrics.items():
                case.report_scalar(tag, float(np.mean(values)))
            case._epoch_accumulated_metrics = None

            if case.epoch == 1 or case.epoch % case.auto_save_each_n_epochs == 0:
                case.checkpoint_save()

    def cache_set(self, keys, values: torch.Tensor, contribution=1.0):
        """
        Cache allows accumulating some information with exponential weighting so
        that it can be retrieved at later iterations.

        Examples: for each image keep its average embedding.
        During retrieval, some of the keys may be missing, so an additional mask is returned.
        Keys is 1-dim iterable that correspond to the first dimension of values tensor.
        """
        assert 0 < contribution <= 1
        if isinstance(keys, pd.Series):
            keys = keys.values.tolist()
        if isinstance(keys, np.ndarray):
            keys = keys.tolist()

        if isinstance(values, torch.Tensor):
            values = self.as_numpy(values)

        for key, val in zip(keys, values, strict=True):
            if key not in self._cache or contribution == 1.0:
                self._cache[key] = val
            else:
                self._cache[key] = self._cache[key] * (1 - contribution) + contribution * val

    def cache_get(self, keys: list) -> tuple:
        """
        Retrieve a couple consisting of mask and a retrieved vector
        """
        mask = []
        result = []
        for key in keys:
            if key in self._cache:
                result.append(self._cache[key])
                mask.append(True)
            else:
                mask.append(False)
        if len(result) == 0:
            # special case
            return None, None
        return self.to_device(np.stack(result)), self.to_device(np.asarray(mask, dtype="bool"))
