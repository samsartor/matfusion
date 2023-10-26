from pathlib import Path
import flax.serialization as ser
import sys

for arg in sys.argv[1:]:
    path_in = Path(arg)
    name_a, name_b = path_in.name.split('_DIFFUSION_', 1)
    path_out = path_in.parent / Path(f'{name_a}_pruned-ema_DIFFUSION_{name_b}')
    print(f'{path_in} -> {path_out}')
    d = ser.msgpack_restore(path_in.read_bytes())
    del d['opt_state']
    del d['params']
    path_out.write_bytes(ser.msgpack_serialize(d))
