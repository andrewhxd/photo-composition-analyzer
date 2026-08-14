#!/usr/bin/env bash
# Downloads and extracts the CADB dataset (~2GB) into data/CADB_Dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
URL='https://www.dropbox.com/scl/fi/fvlsnit7on6218szply4q/CADB_Dataset.zip?rlkey=mwt9eftdhmnawomv44x4deliw&st=8fzb5vej&dl=1'
if [ ! -d data/CADB_Dataset ]; then
  curl -L "$URL" -o data/CADB_Dataset.zip
  unzip -q data/CADB_Dataset.zip -d data
  rm data/CADB_Dataset.zip
fi
echo "Dataset ready: $(ls data/CADB_Dataset/images | wc -l | tr -d ' ') images"
