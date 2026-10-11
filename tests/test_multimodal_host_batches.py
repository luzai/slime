import torch

from slime.backends.megatron_utils import data as data_module


class _Iterator:
    def __init__(self, batch):
        self.batch = batch

    def get_next(self, keys):
        return {key: self.batch[key] for key in keys}


def test_get_batch_moves_host_multimodal_tensors_per_microbatch(monkeypatch):
    moved = []
    original_to = torch.Tensor.to

    def tracking_to(self, *args, **kwargs):
        moved.append(kwargs.get("device"))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(data_module.accelerator, "device", lambda: "cpu")
    monkeypatch.setattr(torch.Tensor, "to", tracking_to)
    pixels = [torch.full((3, 4), float(i)) for i in range(2)]
    grids = [torch.tensor([[1, 1, 3]]) for _ in range(2)]
    batch = dict(
        tokens=[torch.arange(5), torch.arange(4)],
        loss_masks=[torch.ones(2, dtype=torch.int), torch.ones(1, dtype=torch.int)],
        total_lengths=[5, 4],
        response_lengths=[2, 1],
        multimodal_train_inputs=[dict(pixel_values=p, image_grid_thw=g) for p, g in zip(pixels, grids)] + [None],
    )
    out = data_module.get_batch(_Iterator(batch), list(batch), pad_multiplier=1)
    mm = out["multimodal_train_inputs"]
    assert torch.equal(mm["pixel_values"], torch.cat(pixels))
    assert torch.equal(mm["image_grid_thw"], torch.cat(grids))
    assert moved.count("cpu") >= 4  # every multimodal tensor goes through .to(device=current_device())
