import os
from pathlib import Path

import numpy as np


ROW_WIDTH = 256
OUTPUT_SIZE = 64
BYTES_PER_OUTPUT_COLUMN = ROW_WIDTH // OUTPUT_SIZE


def _accumulate_block(
    block,
    valid_bytes,
    first_source_row,
    source_height,
    numerator,
    denominator,
):
    """Add complete source rows, including an optional padded final row."""
    if len(block) % ROW_WIDTH != 0:
        raise ValueError("Block must contain complete source rows.")

    source_rows = len(block) // ROW_WIDTH

    # Each output column covers exactly four source columns.
    values = np.frombuffer(block, dtype=np.uint8)
    values = values.reshape(
        source_rows,
        OUTPUT_SIZE,
        BYTES_PER_OUTPUT_COLUMN,
    )

    row_sums = values.sum(axis=2, dtype=np.int64)

    # Count real bytes, including real bytes whose value is zero.
    row_counts = np.full(
        row_sums.shape,
        BYTES_PER_OUTPUT_COLUMN,
        dtype=np.int64,
    )

    if valid_bytes != len(block):
        valid_in_last_row = valid_bytes - (
            source_rows - 1
        ) * ROW_WIDTH

        if not 1 <= valid_in_last_row < ROW_WIDTH:
            raise ValueError("Only the final row may contain padding.")

        column_starts = (
            np.arange(OUTPUT_SIZE, dtype=np.int64)
            * BYTES_PER_OUTPUT_COLUMN
        )

        row_counts[-1] = np.clip(
            valid_in_last_row - column_starts,
            0,
            BYTES_PER_OUTPUT_COLUMN,
        )

    # Use the same scaled integer coordinates as masked_area_resize.
    # A source row has height OUTPUT_SIZE coordinate units.
    # An output row has height source_height coordinate units.
    first_output_row = (
        first_source_row * OUTPUT_SIZE
    ) // source_height

    last_output_row = (
        (first_source_row + source_rows) * OUTPUT_SIZE - 1
    ) // source_height

    for output_row in range(
        first_output_row,
        last_output_row + 1,
    ):
        top = output_row * source_height
        bottom = (output_row + 1) * source_height

        first_row = max(
            first_source_row,
            top // OUTPUT_SIZE,
        )
        stop_row = min(
            first_source_row + source_rows,
            (bottom + OUTPUT_SIZE - 1) // OUTPUT_SIZE,
        )

        row_indices = np.arange(
            first_row,
            stop_row,
            dtype=np.int64,
        )

        overlap = (
            np.minimum(bottom, (row_indices + 1) * OUTPUT_SIZE)
            - np.maximum(top, row_indices * OUTPUT_SIZE)
        )

        local_start = first_row - first_source_row
        local_stop = stop_row - first_source_row
        weights = overlap[:, None]

        numerator[output_row] += (
            row_sums[local_start:local_stop] * weights
        ).sum(axis=0, dtype=np.int64)

        denominator[output_row] += (
            row_counts[local_start:local_stop] * weights
        ).sum(axis=0, dtype=np.int64)


def stream_to_raw_byte(
    stream,
    file_size,
    chunk_size=65536,
):
    """Return a normalized float64 raw-byte image of shape (64, 64).

    The stream must start at the first byte of the input and contain
    exactly file_size bytes. Working arrays are bounded by chunk_size
    and the fixed output dimensions.
    """
    if type(file_size) is not int or file_size <= 0:
        raise ValueError("file_size must be a positive integer.")

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    source_height = (
        file_size + ROW_WIDTH - 1
    ) // ROW_WIDTH

    # Ensure that the integer weighted sums cannot overflow.
    maximum_row_sum = 255 * BYTES_PER_OUTPUT_COLUMN

    if source_height > np.iinfo(np.int64).max // maximum_row_sum:
        raise ValueError("Input exceeds the integer accumulator limit.")

    numerator = np.zeros(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        dtype=np.int64,
    )
    denominator = np.zeros_like(numerator)

    pending = bytearray()
    remaining = file_size
    processed_rows = 0

    while remaining:
        requested = min(chunk_size, remaining)
        chunk = stream.read(requested)

        if not chunk:
            raise ValueError("Input ended before the declared file size.")

        if len(chunk) > requested:
            raise ValueError("Stream returned more bytes than requested.")

        remaining -= len(chunk)
        pending.extend(chunk)

        complete_bytes = (
            len(pending) // ROW_WIDTH
        ) * ROW_WIDTH

        if complete_bytes:
            block = bytes(pending[:complete_bytes])
            del pending[:complete_bytes]

            _accumulate_block(
                block=block,
                valid_bytes=complete_bytes,
                first_source_row=processed_rows,
                source_height=source_height,
                numerator=numerator,
                denominator=denominator,
            )

            processed_rows += complete_bytes // ROW_WIDTH

    if stream.read(1):
        raise ValueError("Input contains more than the declared file size.")

    if pending:
        valid_bytes = len(pending)
        block = bytes(pending).ljust(ROW_WIDTH, b"\x00")

        _accumulate_block(
            block=block,
            valid_bytes=valid_bytes,
            first_source_row=processed_rows,
            source_height=source_height,
            numerator=numerator,
            denominator=denominator,
        )

        processed_rows += 1

    if processed_rows != source_height:
        raise RuntimeError("Unexpected number of processed source rows.")

    image = np.zeros(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        dtype=np.float64,
    )

    # Empty output regions remain zero.
    np.divide(
        numerator,
        denominator,
        out=image,
        where=denominator > 0,
    )
    image /= 255.0

    return image


def file_to_raw_byte(path, chunk_size=65536):
    """Read a binary as data and return its normalized raw-byte image."""
    path = Path(path)

    with path.open("rb") as file:
        before = os.fstat(file.fileno())

        image = stream_to_raw_byte(
            file,
            file_size=before.st_size,
            chunk_size=chunk_size,
        )

        after = os.fstat(file.fileno())

        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError("Input file changed during processing.")

    return image