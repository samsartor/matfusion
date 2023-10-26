#!/usr/bin/env python

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import jax

if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

import flax.serialization as ser
from matfusion_jax.model import ckptr
from matfusion_jax.config import default_modes

if __name__ == '__main__':
    jax.config.update('jax_platform_name', 'cpu')

    parser = argparse.ArgumentParser(description='Convert a legacy Flax checkpoint into a new Orbax checkpoint.')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()

    if args.mode not in default_modes:
        assert False, f'{args.mode} should be one of {list(default_modes.keys())}'

    m = ser.msgpack_restore(args.input.read_bytes())
    ckptr.save(args.output.absolute(), m)
    (args.output / 'mode.txt').write_text(args.mode)
