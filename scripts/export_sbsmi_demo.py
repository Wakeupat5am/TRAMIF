from pathlib import Path

from PIL import Image

from scripts.sbsmi_core import bytes_to_states, states_to_sbsmi


def main():
    # Synthetic input: 111111 | 111111 | 1111
    sample_bytes = bytes([0xFF, 0xFF])

    states = bytes_to_states(sample_bytes, l=6)
    matrix = states_to_sbsmi(states, l=6)

    height = len(matrix)
    width = len(matrix[0])

    # Flatten the matrix row by row into 8-bit pixel values.
    pixel_bytes = bytes(
        value
        for row in matrix
        for value in row
    )

    project_root = Path(__file__).resolve().parents[1]
    output_path = (
        project_root
        / "data"
        / "derived"
        / "sbsmi_demo"
        / "partial_four_bits.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # L means one grayscale channel, 8 bits per pixel.
    with Image.frombytes("L", (width, height), pixel_bytes) as image:
        image.save(output_path, format="PNG")

    # Read the saved file and verify the decoded pixels.
    with Image.open(output_path) as saved:
        if saved.format != "PNG":
            raise RuntimeError("Unexpected image format.")

        if saved.mode != "L" or saved.size != (width, height):
            raise RuntimeError("Image mode or dimensions changed.")

        if saved.tobytes() != pixel_bytes:
            raise RuntimeError("Pixel values changed after saving.")

        print("States:", states)
        print("Image size:", saved.size)
        print("Image mode:", saved.mode)
        print("Pixel (x=15, y=63):", saved.getpixel((15, 63)))
        print("Pixel (x=63, y=63):", saved.getpixel((63, 63)))
        print("Pixel round-trip: PASS")

    print("Saved to:", output_path.relative_to(project_root))


if __name__ == "__main__":
    main()