"""Frozen prefix TRAIN adapter. No discovery, filtering, inference or training.

Production trust anchors are code constants, not caller-provided checksums.
Only E1/E2 TRAIN metadata and ID-derived train image/report paths are read.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
G2 = "data/derived/historical_episode_feasibility/20261014_233013_930021"
VIEWS = ("sbsmi", "raw-byte", "entropy")
SOURCE_MAPPING_HASH = "a38b074ce5b977db8e7139b7539fc8f290cb970b7b4b7279edaad150eef007be"
HISTORICAL_MANIFEST_HASH = "757251a8986c8808c304d980ff534c4cb8c557f54fcfe3ff168947b5808e362d"
FILES = ("train_keep.csv", "class_mapping.json", "episode_spec.json", "train_00_members.csv")
EPISODES = {
    "E1": (7034, 32, "2019-11", (
        "9743b0cc00278fc415707b61bdd0d060b333aec0b1a264f54ee8e53bdc82b706",
        "ef1019d8a936093c87ecf73b6bfa760cc0718e13aee8e2a885f089c3964dd887",
        "53193445036aa60fb89e2bace637e5287ffcaac36393fd065e7a5ad67cfbb500",
        "16db61fd7b8a8c887e5e05b9c1e539ca6adef7679e7f507b5c761102d25854e7")),
    "E2": (9803, 42, "2019-12", (
        "6c8b1f79449332a82438b13d03fd9bfa9ce0509b3e9017105eecb4e009810768",
        "c5f246b054dbfa9138b63175dc9fc01e257350050b1cab16884307111ddb20d8",
        "c1a060746a3a4bfe5c45de7abdc760a3e6fcb2c4e2aae76aef133bfe234bc326",
        "fb93ebdf873e09f022d9025e7e856194e21cd21d5aac5edaab7e92cfc3200e56")),
}
STORAGE = {
    "sbsmi": ("sbsmi_dataset_v1", "c31fdd58c5fd43ee60de94d5ac63680486e034d1bfd8fffdb01484306bc3f81f"),
    "raw-byte": ("raw_byte_dataset_v1", "2fcc9c692940d8661060ff94ba4989d57972a9443d32a4775641da6baa23312a"),
    "entropy": ("entropy_dataset_v1", "51f3385ad944de1285d4a97ff78d92527fffa52376f0d3bcfe412ac36a915149"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def read_bound(root, relative, expected=None):
    """No symlink/junction redirection; hash and parse the same bytes."""
    root = Path(root).resolve(strict=True)
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_relative_to(root):
        raise ValueError("Path escapes its literal allowlisted location")
    data = path.read_bytes()
    if expected is not None and sha256(data) != expected:
        raise ValueError(f"Checksum mismatch: {relative}")
    return data


def require_hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA-256 identifier")


def validate_mapping(mapping, count):
    if (not isinstance(mapping, dict) or len(mapping) != count
            or any(not isinstance(k, str) or not k for k in mapping)
            or any(type(v) is not int for v in mapping.values())
            or sorted(mapping.values()) != list(range(count))):
        raise ValueError("Invalid contiguous class mapping")


def csv_rows(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Invalid CSV header")
    rows = list(reader)
    if any(None in r or None in r.values() for r in rows):
        raise ValueError("Malformed CSV row")
    return rows


@dataclass(frozen=True)
class TrainingSample:
    sha: str
    timestamp: str
    family: str
    target: int
    source_class_index: int
    content_sha256: str
    group_id: str


def validate_train_rows(rows, members, mapping, source_mapping, count, cutoff):
    """Pure validation also exercised with synthetic fixtures; never filters."""
    start = datetime(2019, 8, 1, tzinfo=timezone.utc)
    end = datetime.fromisoformat(cutoff + "-01T00:00:00+00:00")

    def identity(row):
        require_hash(row["sha"])
        observed = datetime.fromisoformat(row["timestamp"])
        if observed.tzinfo is None or observed.utcoffset().total_seconds() != 0:
            raise ValueError("Expected explicit UTC timestamp")
        if not start <= observed < end or row["stage"] != "train":
            raise ValueError("Non-TRAIN sample or chronology violation")
        if row["family"] not in mapping or row["family"] not in source_mapping:
            raise ValueError("Unknown episode/source family")

    membership = {}
    for member in members:
        identity(member)
        require_hash(member["content_sha256"])
        require_hash(member["group_id"])
        if member["sha"] in membership or member["retained"] not in ("True", "False"):
            raise ValueError("Invalid/duplicate membership")
        membership[member["sha"]] = member
    if len(rows) != count or len({r["sha"] for r in rows}) != count:
        raise ValueError("Wrong TRAIN count or duplicate ID")
    if {r["sha"] for r in rows} != {s for s, m in membership.items() if m["retained"] == "True"}:
        raise ValueError("Retention selection mismatch")
    result = []
    for row in rows:
        identity(row)
        if row["threshold"] != "0" or row["decision"] != "KEEP" or row["reason"] != "retained":
            raise ValueError("Unexpected retention policy")
        if any(row[k] for k in ("reference_sha", "reference_stage", "tlsh_distance")):
            raise ValueError("Retained row unexpectedly has exclusion reference")
        member = membership[row["sha"]]
        if any(row[k] != member[k] for k in ("family", "timestamp", "stage")):
            raise ValueError("TRAIN membership metadata mismatch")
        result.append(TrainingSample(row["sha"], row["timestamp"], row["family"],
                                     mapping[row["family"]], source_mapping[row["family"]],
                                     member["content_sha256"], member["group_id"]))
    if {r.family for r in result} != set(mapping):
        raise ValueError("Missing TRAIN class")
    return tuple(result)


def decode_image(data, report, view):
    """Decode cached images only, preserving legacy scaling exactly."""
    if view not in VIEWS:
        raise ValueError("Unknown view")
    key = "png_sha256" if view == "sbsmi" else "npy_sha256"
    require_hash(report[key])
    if sha256(data) != report[key]:
        raise ValueError("Image checksum mismatch")
    if view == "sbsmi":
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (64, 64):
                raise ValueError("Expected PNG L64x64")
            pixels = np.array(image, dtype=np.uint8, copy=True)
        require_hash(report["pixel_sha256"])
        if sha256(pixels.tobytes()) != report["pixel_sha256"]:
            raise ValueError("Pixel checksum mismatch")
        matrix = pixels.astype(np.float32) / np.float32(255)
    else:
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
        if not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype("<f4"):
            raise ValueError("Expected float32 NPY")
    if matrix.shape != (64, 64) or not np.isfinite(matrix).all():
        raise ValueError("Invalid image shape/finite values")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Image outside [0,1]")
    return torch.from_numpy(np.array(matrix, dtype=np.float32, order="C", copy=True)).unsqueeze(0)


class EpisodeTrainingDataset(Dataset):
    def __init__(self, case, view, project_root=ROOT, image_receipts=None):
        if case not in EPISODES or view not in VIEWS:
            raise ValueError("Only E1/E2 and the three frozen views are supported")
        self.case, self.view = case, view
        self.root = Path(project_root).resolve(strict=True)
        count, self.num_classes, cutoff, hashes = EPISODES[case]
        inputs = {f: read_bound(self.root, f"{G2}/{case}/{f}", h)
                  for f, h in zip(FILES, hashes)}
        self.input_hashes = dict(zip(FILES, hashes))
        spec = parse_json(inputs["episode_spec.json"])
        if spec["train_end"] != cutoff:
            raise ValueError("Episode specification cutoff mismatch")
        self.class_to_idx = parse_json(inputs["class_mapping.json"])
        validate_mapping(self.class_to_idx, self.num_classes)
        source = parse_json(read_bound(self.root, "docs/class_mapping.json", SOURCE_MAPPING_HASH))
        if source["num_classes"] != 51:
            raise ValueError("Source mapping is not Final51")
        validate_mapping(source["class_to_idx"], 51)
        self.samples = validate_train_rows(csv_rows(inputs["train_keep.csv"]),
            csv_rows(inputs["train_00_members.csv"]), self.class_to_idx,
            source["class_to_idx"], count, cutoff)
        folder, config_hash = STORAGE[view]
        self.storage_dir = f"data/derived/{folder}"
        config = parse_json(read_bound(self.root, f"{self.storage_dir}/config.json", config_hash))
        if (config["class_mapping_sha256"] != SOURCE_MAPPING_HASH
                or config["manifest_sha256"] != HISTORICAL_MANIFEST_HASH):
            raise ValueError("Preprocessing source binding mismatch")
        # Same canonicalization used by existing image builders (no trailing LF).
        self.config_id = sha256(json.dumps(config, sort_keys=True).encode("utf-8"))
        self.binding = {"case": case, "view": view, "input_hashes": self.input_hashes,
            "source_mapping_sha256": SOURCE_MAPPING_HASH, "config_file_sha256": config_hash,
            "config_id": self.config_id, "num_classes": self.num_classes,
            "samples": count, "cutoff_exclusive": cutoff,
            "order_sha256": sha256(json_bytes([s.sha for s in self.samples])),
            "class_to_idx": dict(self.class_to_idx)}
        self.image_receipts = None
        if image_receipts is not None:
            if len(image_receipts) != count or [r["sha"] for r in image_receipts] != [s.sha for s in self.samples]:
                raise ValueError("Image receipt ID/order mismatch")
            self.image_receipts = tuple(dict(r) for r in image_receipts)

    def __len__(self):
        return len(self.samples)

    def read_sample(self, index):
        sample = self.samples[index]
        base = f"{self.storage_dir}/train/{sample.sha}"
        report_bytes = read_bound(self.root, base + ".json")
        report = parse_json(report_bytes)
        if not isinstance(report, dict) or type(report.get("class_index")) is not int:
            raise ValueError("Malformed report/source class index")
        expected = {"sample_identifier": sample.sha, "split": "train",
            "timestamp": sample.timestamp, "family": sample.family,
            "class_index": sample.source_class_index, "config_sha256": self.config_id,
            "status": "PASS", "disarmed_content_sha256": sample.content_sha256}
        if self.view == "sbsmi":
            expected["pixel_round_trip"] = "PASS"
        else:
            expected.update(representation="raw-byte" if self.view == "raw-byte" else "local-entropy",
                            image_shape=[64, 64], storage_dtype="<f4", array_round_trip="PASS")
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Report provenance mismatch: {sample.sha}")
        suffix = ".png" if self.view == "sbsmi" else ".npy"
        data = read_bound(self.root, base + suffix)
        receipt = {"sha": sample.sha, "report_sha256": sha256(report_bytes),
                   "image_sha256": sha256(data), "content_sha256": sample.content_sha256}
        if self.image_receipts is not None and receipt != self.image_receipts[index]:
            raise ValueError("Image/report changed since preflight")
        tensor = decode_image(data, report, self.view)
        return tensor, sample.target, sample.sha, receipt

    def __getitem__(self, index):
        tensor, target, sha, _ = self.read_sample(index)
        return tensor, target, sha


def preflight(case, project_root=ROOT):
    """Explicit full image read; never called by constructor/import or tests on real data."""
    from scripts.train_prefix_view import source_hashes
    code = source_hashes(project_root)
    datasets = [EpisodeTrainingDataset(case, v, project_root) for v in VIEWS]
    if any(d.samples != datasets[0].samples for d in datasets[1:]):
        raise ValueError("Cross-view IDs/order/targets/content mismatch")
    output = {"schema": 1, "purpose": "prefix_train_image_preflight", "case": case,
              "code_sha256": code, "views": {}}
    for dataset in datasets:
        receipts = [dataset.read_sample(i)[3] for i in range(len(dataset))]
        output["views"][dataset.view] = {"binding": dataset.binding, "images": receipts}
    output["status"] = "PASS"
    if source_hashes(project_root) != code:
        raise ValueError("Source changed during preflight")
    return output


def main():
    parser = argparse.ArgumentParser(description="Read-only prefix TRAIN image/report preflight; no CNN")
    parser.add_argument("--case", choices=tuple(EPISODES), required=True)
    args = parser.parse_args()
    from scripts.train_prefix_view import write_new_json
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "data/derived/prefix_preflight_v1" / args.case / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipt = preflight(args.case)
        path = output / "receipt.json"
        write_new_json(path, receipt)
        print(f"PASS: {args.case}, three views. Receipt: {path}")
        print(f"SHA-256: {sha256(path.read_bytes())}")
    except BaseException as exc:
        write_new_json(output / "FAILED.json", {"error": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
