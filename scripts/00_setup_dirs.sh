
---

### Scripts

`scripts/00_setup_dirs.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw/exoplanets data/derived docs/logbooks runs logs
echo "OK"
