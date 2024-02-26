# Arguments: $1: GPU ID, $2: optional, offline mode (1: offline, 0: online)

if [ -z "$1" ]
then
    echo "Usage: source env.sh <GPU ID> [offline]"
    return
fi

source venv/bin/activate

# function load_env to export environment variables from function argument file
function load_env() {
    if [ -f "$1" ]; then
        export $(cat $1 | xargs)
    fi
}

echo "Using GPU $1"
export CUDA_VISIBLE_DEVICES=$1

load_env "env/cache_dir.env"

# if pip installed hf_transfer, enable it
if pip list | grep "hf_transfer"; then
    echo "hf_transfer is installed, enabling it"
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# set offline mode
if [ -z "$2" ] || [ "$2" -eq "0" ];  # empty or 0
then
    echo "Online mode"
    load_env "env/online.env"
else
    echo "Offline mode"
    load_env "env/offline.env"
fi