# Source this file in your shell to get environment variables set up.
. ~/PyRSPthon/bin/activate

# Script folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="$DIR:$PYTHONPATH"

