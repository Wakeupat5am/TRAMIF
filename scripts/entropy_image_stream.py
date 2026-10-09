import os
from pathlib import Path

import numpy as np

from scripts.entropy_stream import iter_local_entropy


ROW_WIDTH = 256
OUTPUT_SIZE = 64
VALUES_PER_COLUMN = ROW_WIDTH // OUTPUT_SIZE

WINDOW_SIZE = 256
STRIDE = 128

BUFFER_ROWS = 256
BUFFER_VALUES = BUFFER_ROWS * ROW_WIDTH


def _add_rows(
    values,
    valid_count,
    first_source_row,
    source_height,
    numerator,
    denominator,
):
    """Accumulate complete layout rows, including optional final padding."""
    source_rows = values.size // ROW_WIDTH

    if values.size % ROW_WIDTH != 0:
        raise ValueError("Expected complete layout rows.")

    if not 0 < valid_count <= values.size:
        raise ValueError("Invalid number of real values.")

    grouped = values.reshape(
        source_rows,
        OUTPUT_SIZE,
        VALUES_PER_COLUMN,
    )
    row_sums = grouped.sum(axis=2, dtype=np.float64)
    row_counts = np.full(
        row_sums.shape,
        VALUES_PER_COLUMN,
        dtype=np.int64,
    )

    if valid_count != values.size:
        valid_last_row = valid_count - (source_rows - 1) * ROW_WIDTH

        if not 1 <= valid_last_row < ROW_WIDTH:
            raise ValueError("Only the last row may contain padding.")

        column_starts = (
            np.arange(OUTPUT_SIZE, dtype=np.int64)
            * VALUES_PER_COLUMN
        )
        row_counts[-1] = np.clip(
            valid_last_row - column_starts,
            0,
            VALUES_PER_COLUMN,
        )

    first_output_row = (
        first_source_row * OUTPUT_SIZE
    ) // source_height

    last_output_row = (
        (first_source_row + source_rows) * OUTPUT_SIZE - 1
    ) // source_height

    for output_row in range(first_output_row, last_output_row + 1):
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
        overlaps = (
            np.minimum(bottom, (row_indices + 1) * OUTPUT_SIZE)
            - np.maximum(top, row_indices * OUTPUT_SIZE)
        )

        local_start = first_row - first_source_row
        local_stop = stop_row - first_source_row
        weights = overlaps[:, None]

        # Along x, every output column covers four complete source cells.
        # Their common horizontal area factor cancels in the final ratio.
        numerator[output_row] += (
            row_sums[local_start:local_stop] * weights
        ).sum(axis=0, dtype=np.float64)

        denominator[output_row] += (
            row_counts[local_start:local_stop] * weights
        ).sum(axis=0, dtype=np.int64)


def stream_to_entropy_image(stream, file_size, chunk_size=65536):
    """Return a normalized float64 entropy image of shape (64, 64).

    Configuration:
    - entropy window: 256 bytes
    - stride: 128 bytes
    - source layout width: 256
    - resize: mask-aware area average
    - output range: [0, 1]

    Working storage does not grow with the total file size.
    """
    for name, value in (
        ("file_size", file_size),
        ("chunk_size", chunk_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    source_height = (file_size + ROW_WIDTH - 1) // ROW_WIDTH

    # Protect scaled coordinates and integer area calculations.
    if source_height > np.iinfo(np.int64).max // (2 * OUTPUT_SIZE):
        raise ValueError("Input exceeds the coordinate limit.")

    numerator = np.zeros(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        dtype=np.float64,
    )
    denominator = np.zeros(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        dtype=np.int64,
    )

    buffer = np.empty(BUFFER_VALUES, dtype=np.float64)
    filled = 0
    received = 0
    processed_rows = 0

    blocks = iter_local_entropy(
        stream,
        file_size=file_size,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        chunk_size=chunk_size,
    )

    # Consume the iterator completely, including its final input-size check.
    for block in blocks:
        received += block.size
        position = 0

        while position < block.size:
            take = min(BUFFER_VALUES - filled, block.size - position)
            buffer[filled:filled + take] = block[position:position + take]

            filled += take
            position += take

            if filled == BUFFER_VALUES:
                _add_rows(
                    values=buffer,
                    valid_count=filled,
                    first_source_row=processed_rows,
                    source_height=source_height,
                    numerator=numerator,
                    denominator=denominator,
                )
                processed_rows += BUFFER_ROWS
                filled = 0

    if received != file_size:
        raise RuntimeError("Expected one entropy value per real byte.")

    if filled:
        final_rows = (filled + ROW_WIDTH - 1) // ROW_WIDTH
        padded_count = final_rows * ROW_WIDTH

        # Padding contributes neither values nor valid area.
        buffer[filled:padded_count] = 0.0

        _add_rows(
            values=buffer[:padded_count],
            valid_count=filled,
            first_source_row=processed_rows,
            source_height=source_height,
            numerator=numerator,
            denominator=denominator,
        )
        processed_rows += final_rows

    if processed_rows != source_height:
        raise RuntimeError("Unexpected number of layout rows.")

    image = np.zeros_like(numerator)
    np.divide(
        numerator,
        denominator,
        out=image,
        where=denominator > 0,
    )

    if not np.isfinite(image).all():
        raise RuntimeError("Entropy image contains non-finite values.")

    if (image < 0).any() or (image > 1).any():
        raise RuntimeError("Entropy image is outside [0, 1].")

    return image


def file_to_entropy_image(path, chunk_size=65536):
    """Read a binary as data and return its entropy image."""
    path = Path(path)

    with path.open("rb") as file:
        before = os.fstat(file.fileno())

        image = stream_to_entropy_image(
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