# Bitmap to EPS Converter for Vinyl Cutting

A command line utility that converts bitmap images (PNG, JPG, etc.) to EPS files optimized for vinyl cutting. Each pixel from the input is rendered as a square in the output, with optimized vector paths to minimize duplicate cutting.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python bitmap2eps.py input.png
python bitmap2eps.py input.png --output output.eps
python bitmap2eps.py input1.png input2.png  # Process multiple files
```

## Features

- Converts bitmap images to EPS vector files
- Each pixel is rendered as a square
- Optimized vector paths to prevent double-cutting
- Supports multiple input files
- Customizable output file names 