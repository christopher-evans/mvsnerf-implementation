#!/usr/bin/env bash

# add src to PYTHONPATH
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# set venv
VENV_NAME="venv-mvsnerf"
VENV_LOCATION="../venv-mvsnerf"
python_bin=$(which python)
echo "Using python $python_bin"
#if [[ $python_bin != *VENV_NAME* ]]; then
#  echo "Expected venv not found, setting to $VENV_LOCATION"
#  source "$VENV_LOCATION/bin/activate"
#fi

# call CLI tool
CLI_LOCATION="src/scripts/cli.py"
python $CLI_LOCATION "$@"
