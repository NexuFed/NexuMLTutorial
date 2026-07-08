# NexuMLTutorial

## Create venv and setup NexuML
```bash
uv venv --python 3.13
source .venv/bin/activate
```

### Install dependencies
```bash
uv pip install "git+https://github.com/NexuFed/NexuML.git"[all,dev]

# Or install a branch
uv pip install "git+https://github.com/NexuFed/NexuML.git@feature/nex-187-enhance-the-torch-package-export-to-work-with-nexufl"[all,dev]

# So you can focus only on your own library
uv pip uninstall nexuml_library
```

### Set default paths
```bash
export NEXUML_DATA_ROOT=$(pwd)/data
export NEXUML_LOGS_ROOT=$(pwd)/logs

echo $NEXUML_DATA_ROOT
echo $NEXUML_LOGS_ROOT
```

## Add your own library
Add the library as folder
```bash
nexuml library add $(pwd)/library
nexuml library list
```

### List library components
```bash
nexuml registry --help

nexuml registry list data
nexuml registry list layers
nexuml registry list eval
nexuml registry list scenarios
```

## MNIST ResNet Example
```bash
nexuml train mnist-resnet --max-epochs 1

nexuml tune --scenario-file library/config/tune/mnist_resnet.py --n-trials 10
```