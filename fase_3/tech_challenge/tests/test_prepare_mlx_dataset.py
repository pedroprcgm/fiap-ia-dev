import json

from src.llm.fine_tuning.prepare_mlx_dataset import convert, to_completion_pair


def test_to_completion_pair_combines_instruction_and_input():
    example = {
        "instruction": "Responda com base no protocolo interno.",
        "input": "Qual a conduta?",
        "output": "A conduta e X.",
    }
    pair = to_completion_pair(example)
    assert pair == {
        "prompt": "Responda com base no protocolo interno.\n\nQual a conduta?",
        "completion": "A conduta e X.",
    }


def test_to_completion_pair_without_input_uses_instruction_only():
    example = {"instruction": "Faca algo.", "input": "", "output": "Feito."}
    pair = to_completion_pair(example)
    assert pair["prompt"] == "Faca algo."


def test_convert_writes_train_and_valid_jsonl(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    examples = [
        {"instruction": "I", "input": f"pergunta {i}", "output": f"resposta {i}"}
        for i in range(10)
    ]
    with open(dataset_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    output_dir = tmp_path / "mlx_finetune"
    convert(dataset_path=dataset_path, output_dir=output_dir, valid_fraction=0.2, seed=1)

    train_lines = (output_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    valid_lines = (output_dir / "valid.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(train_lines) + len(valid_lines) == len(examples)
    assert len(valid_lines) >= 2

    for line in train_lines + valid_lines:
        record = json.loads(line)
        assert set(record.keys()) == {"prompt", "completion"}


def test_convert_raises_on_empty_dataset(tmp_path):
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    import pytest

    with pytest.raises(RuntimeError):
        convert(dataset_path=dataset_path, output_dir=tmp_path / "out")
