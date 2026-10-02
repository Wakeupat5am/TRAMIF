import csv
import ctypes
import json
import sys
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.run_preprocessing_pilot import process_sample


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def read_process_memory():
    if sys.platform != "win32":
        raise RuntimeError("This measurement script requires Windows.")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)

    success = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )

    if not success:
        raise ctypes.WinError(ctypes.get_last_error())

    return {
        "working_set_bytes": counters.WorkingSetSize,
        "peak_working_set_bytes": counters.PeakWorkingSetSize,
        "commit_bytes": counters.PagefileUsage,
        "peak_commit_bytes": counters.PeakPagefileUsage,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        project_root
        / "data"
        / "manifests"
        / "train_validation_zip_index.csv"
    )

    min_file_bytes = 5 * 1024 * 1024
    max_file_bytes = 10 * 1024 * 1024

    # Select only train samples above 5 MiB and at most 10 MiB.
    with manifest_path.open(encoding="utf-8", newline="") as file:
        samples = [
            row
            for row in csv.DictReader(file)
            if row["split"] == "train"
            and min_file_bytes < int(row["file_size"]) <= max_file_bytes
        ]

    if not samples:
        raise ValueError("No training samples above 5 and at most 10 MiB.")

    # Choose the largest eligible file, breaking ties by SHA.
    sample = max(
        samples,
        key=lambda row: (int(row["file_size"]), row["sha"]),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        project_root / "data" / "derived" / "memory_pilot" / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    before = read_process_memory()

    print("Selection: train, above 5 MiB and at most 10 MiB", flush=True)
    print("Family:", sample["family"], flush=True)
    print("Sample identifier:", sample["sha"], flush=True)
    print(
        "File size MiB:",
        round(int(sample["file_size"]) / (1024 ** 2), 3),
        flush=True,
    )
    print("Processing one sample with all three views...", flush=True)

    timings = {}
    status = "ERROR"
    error_text = ""
    start = perf_counter()

    try:
        process_sample(
            sample,
            project_root,
            run_dir,
            timings,
            max_file_bytes=max_file_bytes,
        )
        status = "PASS"
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"

    timings["total_seconds"] = perf_counter() - start
    after = read_process_memory()

    report = {
        "sample": sample,
        "selection": {
            "split": "train",
            "minimum_file_bytes_exclusive": min_file_bytes,
            "maximum_file_bytes_inclusive": max_file_bytes,
            "rule": "largest eligible file, ties broken by SHA",
        },
        "status": status,
        "error": error_text,
        "python": sys.version,
        "timings_seconds": timings,
        "memory_before_processing": before,
        "memory_after_processing": after,
        "measurement_scope": (
            "Windows counters for this Python process. "
            "Peaks cover process lifetime, including imports and setup; "
            "they are not per-view peaks."
        ),
    }

    report_path = run_dir / "memory_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    mib = 1024 ** 2

    print("Status:", status)
    print("Total seconds:", round(timings["total_seconds"], 3))
    print(
        "Working set before MiB:",
        round(before["working_set_bytes"] / mib, 2),
    )
    print(
        "Peak working set MiB:",
        round(after["peak_working_set_bytes"] / mib, 2),
    )
    print(
        "Peak commit MiB:",
        round(after["peak_commit_bytes"] / mib, 2),
    )
    print("Report:", report_path)

    if error_text:
        print("Error:", error_text)
        raise SystemExit(1)


if __name__ == "__main__":
    main()