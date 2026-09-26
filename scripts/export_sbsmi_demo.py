from pathlib import Path

from PIL import Image

from scripts.sbsmi_stream import file_to_sbsmi


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "samples" / "sbsmi_toy.bin"

    # Read the file in chunks and directly count transitions.
    chunk_size = 65536
    matrix = file_to_sbsmi(
        input_path,
        l=6,
        chunk_size=chunk_size,
    )

    height = len(matrix)
    width = len(matrix[0])

    # Flatten the matrix row by row into 8-bit pixel values.
    pixel_bytes = bytes(
        value
        for row in matrix
        for value in row
    )

    output_path = (
        project_root
        / "data"
        / "derived"
        / "sbsmi_demo"
        / "partial_four_bits.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.frombytes("L", (width, height), pixel_bytes) as image:
        image.save(output_path, format="PNG")

    # Verify every decoded pixel after saving.
    with Image.open(output_path) as saved:
        if saved.format != "PNG":
            raise RuntimeError("Unexpected image format.")

        if saved.mode != "L" or saved.size != (width, height):
            raise RuntimeError("Image mode or dimensions changed.")

        if saved.tobytes() != pixel_bytes:
            raise RuntimeError("Pixel values changed after saving.")

        print("Input file:", input_path.name)
        print("Input bytes:", input_path.stat().st_size)
        print("Chunk size:", chunk_size)
        print("Image size:", saved.size)
        print("Image mode:", saved.mode)
        print("Pixel (x=15, y=63):", saved.getpixel((15, 63)))
        print("Pixel (x=63, y=63):", saved.getpixel((63, 63)))
        print("Pixel round-trip: PASS")

    print("Saved to:", output_path.relative_to(project_root))


if __name__ == "__main__":
    main()