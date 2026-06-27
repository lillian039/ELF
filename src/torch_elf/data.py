from __future__ import annotations

import json
from typing import Any, Optional, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset, load_from_disk
from torch.utils.data import DataLoader


def build_self_attn_cond_masks(is_cond: Any, is_valid: Any, xp=np):
    encoder_attention_mask = ((is_cond[:, :, None] & is_cond[:, None, :]) | (~is_cond[:, :, None] & is_valid[:, None, :])).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask


def get_pad_token_id(tokenizer: Any, pad_token: str = "pad") -> int:
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return int(token_id)


def pad_and_truncate(ids_list: list[Any], target_len: int, pad_token_id: int):
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def _looks_like_save_to_disk_arrow(ds: Any) -> bool:
    return len(ds) == 1 and any(c.startswith("_") for c in ds.column_names) and not any(not c.startswith("_") for c in ds.column_names)


def load_dataset_split(path: str, dataset_cache_dir=None):
    if path.endswith(".jsonl") or path.endswith(".json"):
        ds = hf_load_dataset("json", data_files=path, split="train", cache_dir=dataset_cache_dir)
        ds.set_format(type="numpy", columns=ds.column_names)
        return ds
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
        ds = ds[splits[0]]
    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir)
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
            ds = ds[splits[0]]
    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_jsonl_dataset(path: str, tokenizer: Any, input_key: str = "input", output_key: str = "output") -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


def load_dataset(config: Any, dataset_cache_dir=None):
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
    return train_dataset, eval_dataset


def prepare_batch(batch: dict[str, Any], config: Any, device: torch.device) -> dict[str, Any]:
    result = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    batch_size = result["input_ids"].shape[0]
    label_drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if config.label_drop_prob > 0:
        label_drop_mask = torch.rand(batch_size, device=device) < config.label_drop_prob
    result["label_drop_mask"] = label_drop_mask
    return result


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0, drop_last: bool = True, max_seq_length: int = 512, pad_token_id: int = 0, max_input_seq_length: Optional[int] = None):
    def collate_fn(batch_list):
        input_ids_list = [np.array(item["input_ids"]) for item in batch_list]
        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.array(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.array(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)
        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        result: dict[str, Any] = {
            "input_ids": torch.from_numpy(ids).long(),
            "encoder_attention_mask": torch.from_numpy(encoder_attn),
            "attention_mask": torch.from_numpy(attn),
            "cond_seq_mask": torch.from_numpy(pred),
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    dataset_items = cast(Any, list(dataset))
    return DataLoader(dataset_items, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, drop_last=drop_last, persistent_workers=num_workers > 0)
