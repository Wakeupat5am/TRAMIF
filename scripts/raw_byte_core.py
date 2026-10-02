def bytes_to_layout(
    data: bytes,
    row_width: int = 256,
) -> tuple[list[list[int]], list[list[int]]]:
    """Arrange bytes into rows and return values plus a validity mask."""

    if not isinstance(row_width, int) or row_width <= 0:
        raise ValueError("row_width must be a positive integer.")

    if not data:
        raise ValueError("Input data must not be empty.")

    # Round up so the final partial row is included.
    height = (len(data) + row_width - 1) // row_width

    values = [
        [0 for _ in range(row_width)]
        for _ in range(height)
    ]
    mask = [
        [0 for _ in range(row_width)]
        for _ in range(height)
    ]

    for offset, value in enumerate(data):
        row, column = divmod(offset, row_width)

        values[row][column] = value
        mask[row][column] = 1

    return values, mask


if __name__ == "__main__":
    sample = bytes([0, 64, 128, 255, 32])
    values, mask = bytes_to_layout(sample, row_width=4)

    print("Input bytes:", list(sample))
    print(f"Layout shape: {len(values)} x {len(values[0])}")

    print("Values:")
    for row in values:
        print(row)

    print("Mask:")
    for row in mask:
        print(row)

    print("Valid pixels:", sum(sum(row) for row in mask))