import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.training.train as train_mod


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.w


def test_trainer_checkpoint_mode_max_uses_larger_is_better(monkeypatch, tmp_path):
    metric_values = iter([0.2, 0.4, 0.3])
    saved = []

    def fake_train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip=None):
        return {"loss": 1.0}

    def fake_evaluate_epoch(model, loader, loss_fn, device):
        return {
            "loss": 1.0,
            "mae": 1.0,
            "rmse": 1.0,
            "rel_l2": next(metric_values),
            "max_error": 1.0,
        }

    def fake_save_checkpoint(path, model, optimizer, epoch, metrics, cfg, **kwargs):
        saved.append((Path(path).name, int(epoch), float(metrics.get("val_rel_l2", -1.0))))

    monkeypatch.setattr(train_mod, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_mod, "evaluate_epoch", fake_evaluate_epoch)
    monkeypatch.setattr(train_mod, "save_checkpoint", fake_save_checkpoint)

    cfg = {
        "output_dir": str(tmp_path / "out"),
        "train": {
            "epochs": 3,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "loss": "mse",
            "checkpoint_metric": "val_rel_l2",
            "early_stopping": {"patience": 10, "mode": "max"},
        },
    }
    loaders = {"train": [object()], "val": [object()]}
    trainer = train_mod.Trainer(TinyModel(), loaders, cfg, device=torch.device("cpu"))
    trainer.fit()

    best_saves = [row for row in saved if row[0] == "best.pt"]
    assert len(best_saves) == 2
    assert best_saves[0][1] == 1 and abs(best_saves[0][2] - 0.2) < 1e-8
    assert best_saves[1][1] == 2 and abs(best_saves[1][2] - 0.4) < 1e-8
