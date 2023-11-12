# Generate pipelines on sampled target sets

## Prerequisites

Requires:
- Python: 3.8.15
- Git LFS: https://git-lfs.com/

Install required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Setup data 

1. Install data from LFS:

```bash
git lfs fetch
git lfs pull
```

2. Unzip data:

```bash
cd app/data/
cat data.tar.gz.* | tar xzvf -
```

## Run script

1. Example of a default-configured script (ATTENTION: this should be run on server, at the root of the project):

```shell
python -m scripts.generate_annotated_pipelines
```

Default arguments for script:

```python
parser.add_argument("--target_sets", type=int, default=200)
parser.add_argument("--policy_episodes", type=int, default=50)
parser.add_argument("--sampling_rate", type=float, default=0.5)
parser.add_argument("--min_item_set_nodes", type=int, default=100)
parser.add_argument("--max_item_set_nodes", type=int, default=10000)
parser.add_argument("--min_pipeline_size", type=int, default=4)
parser.add_argument("--max_pipeline_size", type=int, default=8)
```

2. Example of a custom-configured script (run at the root of the project):

```shell
python -m scripts.generate_annotated_pipelines --target_sets 3 --policy_episodes 1
```
