# Bitmap to EPS Converter for Vinyl Cutting

A command line utility that converts bitmap images (PNG, BMP, etc.) to EPS files optimized for vinyl cutting. Each pixel from the input is rendered as a square in the output, with optimized vector paths to minimize duplicate cutting.

<div style="display: flex; align-items: center;">
  <img src="docs/sample.png" alt="Sample bitmap" width="150" height="150" style="image-rendering: pixelated;">
<span style="display: inline-block;padding: 5px 5px;">&rArr;</span>
<img src="docs/sample.svg" alt="Cutting Vectors" width="160" height="160">
</div>


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python pix2eps.py input.png   # output file name: input.eps
python pix2eps.py input.png --output output.eps
python pix2eps.py input1.png input2.png  # Process multiple files
python pix2eps.py --scale 18 input.png   # Each output square is .25 inches  (assuming the cutter software is set at default 72pt/inch)
python pix2eps.py --help  # see all options
```

## Features

- Converts bitmap images to EPS vector files
- Each pixel is rendered as a square (see the scaling option to change square size)
- Optimized vector paths to reduce double-cutting
- Supports multiple input files
- Areas to remove are marked with a diagonal slash cut

