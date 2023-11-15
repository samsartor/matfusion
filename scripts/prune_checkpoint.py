from pathlib import Path
import flax.serialization as ser
import argparse
from typing import Any, cast

parser = argparse.ArgumentParser()
parser.add_argument('input_checkpoint', type=Path)
parser.add_argument('output_checkpoint', type=Path)
parser.add_argument('-k', '--key', type=str, nargs='*', default=['opt_state', 'params'])
args = parser.parse_args()

in_path: Path = args.input_checkpoint
out_path: Path = args.output_checkpoint
keys: list[str] = args.key

out_path.mkdir(exist_ok=False)

print('pruning', ', '.join(keys))

in_bytes = (in_path / 'checkpoint.msgpack').read_bytes()
d = cast(dict[str, Any], ser.msgpack_restore(in_bytes))
for k in keys:
    del d[k]
out_bytes = ser.msgpack_serialize(d)
(out_path / 'checkpoint.msgpack').write_bytes(out_bytes)

(out_path / 'mode.json').write_text((in_path / 'mode.json').read_text())
