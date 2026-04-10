#!/usr/bin/env python3
"""Generate simple solid-color PNG icons for the AuthentiWrite browser extension.
Uses only struct and zlib -- no Pillow required."""

import struct
import zlib
import os

def create_png(width, height, r, g, b):
    """Create a minimal valid PNG with a solid color."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)

    # IDAT chunk - raw image data
    raw_data = b''
    row = bytes([r, g, b] * width)
    for _ in range(height):
        raw_data += b'\x00' + row  # filter byte 0 (None) + pixel data

    compressed = zlib.compress(raw_data)
    idat = chunk(b'IDAT', compressed)

    # IEND chunk
    iend = chunk(b'IEND', b'')

    return signature + ihdr + idat + iend

def main():
    # Indigo #6366f1
    r, g, b = 0x63, 0x66, 0xf1

    sizes = [16, 48, 128]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(output_dir, exist_ok=True)

    for size in sizes:
        png_data = create_png(size, size, r, g, b)
        filepath = os.path.join(output_dir, f'icon{size}.png')
        with open(filepath, 'wb') as f:
            f.write(png_data)
        print(f'Created {filepath} ({len(png_data)} bytes)')

if __name__ == '__main__':
    main()
