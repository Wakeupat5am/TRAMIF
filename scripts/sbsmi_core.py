def bytes_to_states(data: bytes, l: int = 6) -> list[int]:
    """Read complete l-bit states, most significant bit first."""

    if not isinstance(l, int) or not 1 <= l <= 8:
        raise ValueError("l must be an integer between 1 and 8.")

    states = []
    buffer = 0
    bits_available = 0
    mask = (1 << l) - 1

    for byte in data:
        # Append eight new bits after any bits left from the previous byte.
        buffer = (buffer << 8) | byte
        bits_available += 8

        while bits_available >= l:
            bits_available -= l
            state = (buffer >> bits_available) & mask
            states.append(state)

        # Keep only the bits that have not been consumed.
        buffer &= (1 << bits_available) - 1

    if bits_available:
        raise ValueError(
            f"{bits_available} trailing bits remain; "
            "the incomplete-state policy has not been configured."
        )

    return states

def states_to_sbsmi(states: list[int], l: int = 6) -> list[list[int]]:
    """Convert a sequence of l-bit states into grayscale pixel values."""

    # Implementation limit to keep the matrix small.
    if not isinstance(l, int) or not 1 <= l <= 8:
        raise ValueError("l must be an integer between 1 and 8.")

    if len(states) < 2:
        raise ValueError("At least two states are needed for a transition.")

    size = 2 ** l

    for state in states:
        if not isinstance(state, int) or not 0 <= state < size:
            raise ValueError(f"Each state must be an integer in [0, {size - 1}].")

    # Each row is a separate list.
    counts = [[0 for _ in range(size)] for _ in range(size)]

    # Count transitions between consecutive states.
    for previous, current in zip(states, states[1:]):
        counts[previous][current] += 1

    pixels = []

    for row in counts:
        row_total = sum(row)

        if row_total == 0:
            pixels.append([0] * size)
        else:
            # Integer division computes floor(255 * count / row_total).
            pixels.append([
                count * 255 // row_total
                for count in row
            ])

    return pixels


if __name__ == "__main__":
    example_bytes = bytes([0x00, 0x10, 0x83])

    states = bytes_to_states(example_bytes, l=6)
    image = states_to_sbsmi(states, l=6)

    print("Bytes (hex):", example_bytes.hex(" "))
    print("States:", states)
    print(f"Image shape: {len(image)} x {len(image[0])}")

    print("Nonzero pixels:")
    for previous, row in enumerate(image):
        for current, value in enumerate(row):
            if value != 0:
                print(f"({previous}, {current}): {value}")