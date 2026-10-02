from math import isfinite


def masked_area_resize(
    values: list[list[float]],
    mask: list[list[int]],
    output_height: int = 64,
    output_width: int = 64,
) -> list[list[float]]:
    """Resize using area averages, excluding invalid source cells."""

    for dimension in (output_height, output_width):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("Output dimensions must be positive integers.")

    if not values or not values[0]:
        raise ValueError("Input must be a nonempty matrix.")

    source_height = len(values)
    source_width = len(values[0])

    if len(mask) != source_height:
        raise ValueError("Mask height must match input height.")

    for value_row, mask_row in zip(values, mask):
        if len(value_row) != source_width:
            raise ValueError("Input rows must have equal lengths.")

        if len(mask_row) != source_width:
            raise ValueError("Mask shape must match input shape.")

        for value, valid in zip(value_row, mask_row):
            if valid not in (0, 1):
                raise ValueError("Mask values must be 0 or 1.")

            if not isfinite(value):
                raise ValueError("Input values must be finite.")

    output = []

    # Use scaled integer coordinates to calculate exact overlap areas.
    # Along y, each source cell spans output_height coordinate units.
    # Along x, each source cell spans output_width coordinate units.
    for out_y in range(output_height):
        top = out_y * source_height
        bottom = (out_y + 1) * source_height

        first_y = top // output_height
        stop_y = (
            bottom + output_height - 1
        ) // output_height

        output_row = []

        for out_x in range(output_width):
            left = out_x * source_width
            right = (out_x + 1) * source_width

            first_x = left // output_width
            stop_x = (
                right + output_width - 1
            ) // output_width

            weighted_sum = 0.0
            valid_area = 0

            for source_y in range(first_y, stop_y):
                overlap_y = (
                    min(bottom, (source_y + 1) * output_height)
                    - max(top, source_y * output_height)
                )

                for source_x in range(first_x, stop_x):
                    if mask[source_y][source_x] == 0:
                        continue

                    overlap_x = (
                        min(right, (source_x + 1) * output_width)
                        - max(left, source_x * output_width)
                    )

                    area = overlap_y * overlap_x
                    weighted_sum += values[source_y][source_x] * area
                    valid_area += area

            if valid_area == 0:
                output_row.append(0.0)
            else:
                output_row.append(weighted_sum / valid_area)

        output.append(output_row)

    return output