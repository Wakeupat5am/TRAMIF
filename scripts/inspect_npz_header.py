import ast
import struct
from pathlib import Path
from zipfile import ZipFile


def read_exact(stream, size):
    data = stream.read(size)

    if len(data) != size:
        raise ValueError("Unexpected end of NPY header.")

    return data


def read_npy_header(stream):
    magic = read_exact(stream, 6)

    if magic != b"\x93NUMPY":
        raise ValueError("Invalid NPY signature.")

    version = tuple(read_exact(stream, 2))

    if version == (1, 0):
        header_size = struct.unpack("<H", read_exact(stream, 2))[0]
        encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        header_size = struct.unpack("<I", read_exact(stream, 4))[0]
        encoding = "utf-8" if version == (3, 0) else "latin1"
    else:
        raise ValueError(f"Unsupported NPY version: {version}")

    if header_size > 65536:
        raise ValueError("Header exceeds this inspector's size limit.")

    header_text = read_exact(stream, header_size).decode(encoding)
    header = ast.literal_eval(header_text.strip())

    return version, header


def main():
    project_root = Path(__file__).resolve().parents[1]
    npz_path = project_root / "data" / "raw" / "bodmas.npz"

    with ZipFile(npz_path, "r") as archive:
        for name in ("X.npy", "y.npy"):
            entry = archive.getinfo(name)

            with archive.open(entry, "r") as stream:
                version, header = read_npy_header(stream)

            print("Array:", name)
            print("NPY version:", version)
            print("Shape:", header["shape"])
            print("Dtype descriptor:", header["descr"])
            print("Fortran order:", header["fortran_order"])
            print(f"Uncompressed size: {entry.file_size / (1024 ** 3):.4f} GiB")
            print()


if __name__ == "__main__":
    main()