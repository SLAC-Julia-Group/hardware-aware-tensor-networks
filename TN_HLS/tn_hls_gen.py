#!/usr/bin/env python3
"""
Main script for generating HLS code from tensor network model specification
"""

import argparse
import sys
from pathlib import Path

from tn_hls.config import load_model_config, print_model_summary
from tn_hls.generator import HLSGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate HLS code for tensor network models')
    parser.add_argument('-c', '--config', required=True, help='Model configuration file (YAML)')
    parser.add_argument('-o', '--output', default='tn_model.cpp', help='Output HLS file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading model from {args.config}...")
    try:
        model = load_model_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Show model summary
    if args.verbose:
        print_model_summary(model)
    
    # Generate HLS
    print(f"Generating HLS code...")
    generator = HLSGenerator(model)
    hls_code = generator.generate()
    
    # Write output
    output_path = Path(args.output)
    output_path.write_text(hls_code)
    print(f"✓ Generated {args.output}")


if __name__ == '__main__':
    main()