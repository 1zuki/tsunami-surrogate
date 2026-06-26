import torch

from src.training.metrics import MetricAccumulator, compute_metrics


def _aggregate_batches(batches):
    acc = MetricAccumulator()
    for pred, target in batches:
        acc.update(pred, target)

    return acc.compute()


def test_metric_accumulator_uses_dataset_global_relative_l2():
    batches = [
        (torch.tensor([[[1.0, 2.0]]]), torch.tensor([[[1.0, 4.0]]])),
        (torch.tensor([[[9.0, 9.0]]]), torch.tensor([[[100.0, 100.0]]])),
    ]

    got = _aggregate_batches(batches)
    pred_all = torch.cat([b[0].reshape(-1) for b in batches])
    target_all = torch.cat([b[1].reshape(-1) for b in batches])
    expected = torch.linalg.vector_norm(pred_all - target_all) / (
        torch.linalg.vector_norm(target_all) + 1e-8
    )

    old_batch_mean = sum(
        compute_metrics(pred, target)["rel_l2"] for pred, target in batches
    ) / len(batches)

    torch.testing.assert_close(
        torch.tensor(got["rel_l2"]), expected, rtol=1e-6, atol=1e-8
    )
    assert abs(got["rel_l2"] - old_batch_mean) > 1e-3


def test_metric_accumulator_reports_true_global_max_error():
    batches = [
        (torch.zeros((2, 1, 2)), torch.zeros((2, 1, 2))),
        (torch.tensor([[[0.0, 100.0]]]), torch.zeros((1, 1, 2))),
    ]

    got = _aggregate_batches(batches)
    old_weighted_batch_max = sum(
        compute_metrics(pred, target)["max_error"] * int(pred.shape[0])
        for pred, target in batches
    ) / sum(int(pred.shape[0]) for pred, _ in batches)

    assert got["max_error"] == 100.0
    assert old_weighted_batch_max < got["max_error"]


def test_metric_accumulator_uses_dataset_global_rmse():
    batches = [
        (torch.zeros((3, 1, 2)), torch.zeros((3, 1, 2))),
        (torch.full((1, 1, 2), 4.0), torch.zeros((1, 1, 2))),
    ]

    got = _aggregate_batches(batches)
    pred_all = torch.cat([b[0].reshape(-1) for b in batches])
    target_all = torch.cat([b[1].reshape(-1) for b in batches])
    expected = torch.sqrt(torch.mean((pred_all - target_all) ** 2))

    old_weighted_batch_rmse = sum(
        compute_metrics(pred, target)["rmse"] * int(pred.shape[0])
        for pred, target in batches
    ) / sum(int(pred.shape[0]) for pred, _ in batches)

    torch.testing.assert_close(
        torch.tensor(got["rmse"]), expected, rtol=1e-6, atol=1e-8
    )
    assert got["rmse"] > old_weighted_batch_rmse
