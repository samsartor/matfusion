import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

path = Path(sys.argv[1])

from matfusion_jax.config import default_modes
import json

mode = (path / 'mode.txt').read_text().strip()
(path / 'mode.json').write_text(json.dumps(default_modes[mode], indent=2))
(path / 'mode.txt').unlink()
