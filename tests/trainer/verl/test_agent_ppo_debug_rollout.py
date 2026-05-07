import importlib.util
import pickle
import sys
import types
from pathlib import Path


class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class FakeTensor:
    def __init__(self, data):
        self.data = list(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeTensor(self.data[:])

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.data == other.data


class FakeDataProto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def save_to_disk(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


def _load_agent_ppo_trainer_module():
    verl = types.ModuleType("verl")

    verl.DataProto = FakeDataProto
    sys.modules["verl"] = verl

    verl_protocol = types.ModuleType("verl.protocol")
    verl_protocol.pad_dataproto_to_divisor = lambda batch, divisor: (batch, 0)
    sys.modules["verl.protocol"] = verl_protocol

    verl_ray = types.ModuleType("verl.single_controller.ray")
    verl_ray.RayWorkerGroup = object
    sys.modules["verl.single_controller.ray"] = verl_ray

    verl_core_algos = types.ModuleType("verl.trainer.ppo.core_algos")
    verl_core_algos.agg_loss = lambda **kwargs: None
    sys.modules["verl.trainer.ppo.core_algos"] = verl_core_algos

    verl_metric_utils = types.ModuleType("verl.trainer.ppo.metric_utils")
    verl_metric_utils.compute_data_metrics = lambda **kwargs: {}
    verl_metric_utils.compute_timing_metrics = lambda **kwargs: {}
    sys.modules["verl.trainer.ppo.metric_utils"] = verl_metric_utils

    verl_ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")
    verl_ray_trainer.RayPPOTrainer = object
    verl_ray_trainer.ResourcePoolManager = object
    verl_ray_trainer.compute_advantage = lambda *a, **k: None
    verl_ray_trainer.compute_response_mask = lambda *a, **k: None
    sys.modules["verl.trainer.ppo.ray_trainer"] = verl_ray_trainer

    verl_utils = types.ModuleType("verl.trainer.ppo.utils")
    verl_utils.Role = object
    verl_utils.WorkerType = object
    sys.modules["verl.trainer.ppo.utils"] = verl_utils

    verl_debug = types.ModuleType("verl.utils.debug")
    class _MarkedTimer:
        def __call__(self, *args, **kwargs):
            class _Ctx:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()
    verl_debug.marked_timer = _MarkedTimer()
    sys.modules["verl.utils.debug"] = verl_debug

    verl_metric = types.ModuleType("verl.utils.metric")
    verl_metric.reduce_metrics = lambda x: x
    sys.modules["verl.utils.metric"] = verl_metric

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(to_container=lambda *a, **k: {})
    sys.modules["omegaconf"] = omegaconf

    torch = types.ModuleType("torch")

    def tensor(data, dtype=None):  # noqa: ARG001
        return FakeTensor(data)

    def save(obj, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    def load(filepath, map_location=None, weights_only=False):  # noqa: ARG001
        with open(filepath, "rb") as f:
            return pickle.load(f)

    torch.Tensor = FakeTensor
    torch.tensor = tensor
    torch.save = save
    torch.load = load
    torch.long = "long"
    torch.float32 = "float32"
    sys.modules["torch"] = torch

    sys.modules["torch_npu"] = types.ModuleType("torch_npu")

    rllm_engine = types.ModuleType("rllm.engine.agent_execution_engine")
    rllm_engine.AsyncAgentExecutionEngine = object
    sys.modules["rllm.engine.agent_execution_engine"] = rllm_engine

    module_path = Path(__file__).resolve().parents[3] / "rllm" / "trainer" / "verl" / "agent_ppo_trainer.py"
    spec = importlib.util.spec_from_file_location("rllm_agent_ppo_trainer_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _make_trainer(mod, tmp_path: Path, *, save=False, load_path=None):
    trainer = object.__new__(mod.AgentPPOTrainer)
    trainer.config = AttrDict(
        rllm=AttrDict(
            stepwise_advantage=AttrDict(enable=False),
            debug_rollout=AttrDict(
                save=save,
                load_path=load_path,
                dir=str(tmp_path / "debug_rollouts"),
                filename=None,
            ),
        ),
        trainer=AttrDict(default_local_dir=str(tmp_path)),
    )
    trainer.global_steps = 7
    return trainer


def _make_dataproto(mod):
    return mod.DataProto(
        batch={"prompts": mod.torch.tensor([1, 2]), "responses": mod.torch.tensor([3, 4])},
        meta_info={"foo": "bar"},
    )


def test_debug_rollout_dump_round_trip(tmp_path):
    mod = _load_agent_ppo_trainer_module()
    trainer = _make_trainer(mod, tmp_path, save=True)

    trajectories = [{"prompt_tokens": mod.torch.tensor([1, 2]), "metrics": {"steps": 1}, "idx": 0}]
    batch = _make_dataproto(mod)
    metrics = {"traj/steps_mean": 1.0}

    trainer._save_debug_rollout_dump(trajectories, batch, metrics, global_steps=7, mode="trajectory")
    dump_path = tmp_path / "debug_rollouts" / "rollout_step_7.pt"
    assert dump_path.exists()

    loader = _make_trainer(mod, tmp_path, load_path=str(dump_path))
    loaded_trajectories, loaded_batch, loaded_metrics = loader._load_debug_rollout_dump(global_steps=999, mode="trajectory")

    assert loaded_metrics == metrics
    assert loaded_batch.meta_info == batch.meta_info
    assert loaded_batch.batch["prompts"] == batch.batch["prompts"]
    assert loaded_trajectories[0]["prompt_tokens"] == trajectories[0]["prompt_tokens"]
    assert loaded_trajectories[0]["metrics"] == trajectories[0]["metrics"]


def test_debug_rollout_dump_mode_mismatch_raises(tmp_path):
    mod = _load_agent_ppo_trainer_module()
    trainer = _make_trainer(mod, tmp_path, save=True)
    trainer._save_debug_rollout_dump([], _make_dataproto(mod), {}, global_steps=7, mode="trajectory")
    dump_path = tmp_path / "debug_rollouts" / "rollout_step_7.pt"

    loader = _make_trainer(mod, tmp_path, load_path=str(dump_path))
    try:
        loader._load_debug_rollout_dump(global_steps=7, mode="step")
        assert False, "Expected mode mismatch to raise"
    except ValueError as exc:
        assert "mode mismatch" in str(exc)
