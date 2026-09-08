import json
import pickle

import pytest

from src.llm.models.domain_llm import _detect_backend_kind


def test_detects_retrieval_demo(tmp_path):
    with open(tmp_path / "tfidf_index.pkl", "wb") as f:
        pickle.dump({"vectorizer": None, "tfidf_matrix": None, "answers": []}, f)

    assert _detect_backend_kind(tmp_path) == "retrieval_demo"


def test_detects_hf_peft_adapter_by_weights_file(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"some_key": "value"}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")

    assert _detect_backend_kind(tmp_path) == "lora_hf"


def test_detects_hf_peft_adapter_by_config_key(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "some/model"})
    )

    assert _detect_backend_kind(tmp_path) == "lora_hf"


def test_detects_mlx_adapter_by_weights_file(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"model": "mlx-community/x"}))
    (tmp_path / "adapters.safetensors").write_bytes(b"")

    assert _detect_backend_kind(tmp_path) == "lora_mlx"


def test_detects_mlx_adapter_by_config_key(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"model": "mlx-community/x", "fine_tune_type": "lora", "lora_parameters": {"rank": 8}})
    )

    assert _detect_backend_kind(tmp_path) == "lora_mlx"


def test_unrecognized_adapter_config_raises_value_error(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"something": "else"}))

    with pytest.raises(ValueError):
        _detect_backend_kind(tmp_path)


def test_missing_everything_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _detect_backend_kind(tmp_path)
