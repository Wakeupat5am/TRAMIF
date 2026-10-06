import hashlib
import io
import re
from importlib.metadata import version

import tlsh


# Known-answer distance from the official TLSH regression tests.
REFERENCE_A = (
    "T106D29517F780237185070293B60E36FAB735C0F833D66460688DA22D6756E751B7BAEB"
)
REFERENCE_B = (
    "T1E951784702042376169012B1BA5A76EAF36092FC3311A595B4856235278F9F973763EF"
)
REFERENCE_DISTANCE = 427

# Regression value recorded from our deterministic synthetic input.
SYNTHETIC_HASH = (
    "T14B531294114F70CB3C4BC9DB2A347378B97ACA6134E5085847C476BBA09BA489FBB6C0"
)


def hash_stream(stream, chunk_size=65536):
    """Hash a binary stream, buffering its first five bytes.

    Workaround for the initial short-update behavior observed in
    py-tlsh 5.0.0. No bytes are discarded, duplicated, or padded.
    """
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")

    prefix = bytearray()

    # TLSH must receive the complete initial five-byte window together.
    # The underlying stream can still be read one byte at a time.
    while len(prefix) < 5:
        read_size = min(chunk_size, 5 - len(prefix))
        chunk = stream.read(read_size)

        if not chunk:
            return "TNULL"

        prefix.extend(chunk)

    hasher = tlsh.Tlsh()
    hasher.update(bytes(prefix))
    total_bytes = len(prefix)

    while chunk := stream.read(chunk_size):
        hasher.update(chunk)
        total_bytes += len(chunk)

    # Default TLSH requires at least 50 input bytes.
    if total_bytes < 50:
        return "TNULL"

    hasher.final()

    # Sufficient length alone does not guarantee sufficient variation.
    if not hasher.is_valid:
        return "TNULL"

    return hasher.hexdigest()


def hash_in_chunks(data, chunk_size):
    with io.BytesIO(data) as stream:
        return hash_stream(stream, chunk_size)


def main():
    installed_version = version("py-tlsh")
    print("py-tlsh:", installed_version)

    if installed_version != "5.0.0":
        raise RuntimeError("Expected the pinned py-tlsh version 5.0.0.")

    if tlsh.diff(REFERENCE_A, REFERENCE_B) != REFERENCE_DISTANCE:
        raise RuntimeError("Reference distance mismatch.")

    if tlsh.diff(REFERENCE_B, REFERENCE_A) != REFERENCE_DISTANCE:
        raise RuntimeError("Reverse reference distance mismatch.")

    if tlsh.diff(REFERENCE_A, REFERENCE_A) != 0:
        raise RuntimeError("Self-distance must be zero.")

    print("Reference distance: 427 — PASS")
    print("Symmetry and self-distance: PASS")

    # Deterministic synthetic bytes. No dataset files are accessed.
    data = b"".join(
        hashlib.sha256(index.to_bytes(4, "big")).digest()
        for index in range(2048)
    ) + b"tail-end"

    expected_hash = tlsh.hash(data)

    if re.fullmatch(r"T1[0-9A-F]{70}", expected_hash) is None:
        raise RuntimeError(f"Unexpected TLSH digest: {expected_hash}")

    if expected_hash != SYNTHETIC_HASH:
        raise RuntimeError("Synthetic one-shot regression hash changed.")

    print("Synthetic input bytes:", len(data))
    print("One-shot regression hash: PASS")
    print("Streaming policy: buffer the first 5 bytes")

    # Include small reads and final tails of 1, 2, 3, and 4 bytes
    # after a 65536-byte read following the five-byte prefix.
    lengths = (
        50,
        51,
        257,
        65536,
        65542,
        65543,
        65544,
        65545,
    )
    chunk_sizes = (1, 2, 3, 4, 5, 7, 512, 65536)

    # Extend the fixture so every requested length is available.
    test_data = data + b"extra-test-bytes"
    checks = 0

    for chunk_size in chunk_sizes:
        for length in lengths:
            sample = test_data[:length]
            expected = tlsh.hash(sample)
            actual = hash_in_chunks(sample, chunk_size)

            if actual != expected:
                raise RuntimeError(
                    "Streaming hash mismatch: "
                    f"input_bytes={length}, chunk_size={chunk_size}, "
                    f"expected={expected}, actual={actual}"
                )

            checks += 1

        print(
            f"Chunk size {chunk_size}: "
            f"{len(lengths)} input lengths — PASS"
        )

    print("Exact streaming comparisons:", checks)

    invalid_inputs = (
        b"",
        b"abc",
        data[:49],
        b"\x00" * 4096,
    )

    for sample in invalid_inputs:
        if tlsh.hash(sample) != "TNULL":
            raise RuntimeError("Expected one-shot TNULL.")

        for chunk_size in chunk_sizes:
            if hash_in_chunks(sample, chunk_size) != "TNULL":
                raise RuntimeError(
                    "Expected streaming TNULL: "
                    f"input_bytes={len(sample)}, "
                    f"chunk_size={chunk_size}"
                )

    print("Short and constant-byte inputs: TNULL — PASS")
    print("TLSH buffered streaming checks: PASS")
    print("No dataset files accessed or changed.")


if __name__ == "__main__":
    main()