#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set -eo pipefail
conda activate myenv
#pip install pylas pptk pyransac3d
#pip install -e ./
#exec python setup.py install
export FLASK_APP=run.py
exec flask run --host=0.0.0.0
