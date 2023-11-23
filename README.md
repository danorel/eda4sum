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

## Setup environment variables

1. Create `.env` file based on [environment variables](./.env.example) (should be stored at the root of the project):

```
WANDB_PROJECT=xeda
WANDB_API_KEY=
```

IMPORTANT! If you are running on a server and don't want monitoring to be recorded keep `WANDB_API_KEY` empty.

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
parser.add_argument("--skip_node_sampling", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--skip_train_policies", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--skip_generate_pipelines", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--skip_annotate_pipelines", action=argparse.BooleanOptionalAction, default=False)
```

2. Example of a custom-configured script (run at the root of the project):

```shell
python -m scripts.generate_annotated_pipelines --target_sets 3 --policy_episodes 1
```

3. Example of a partial script (run at the root of the project):

```shell
python -m scripts.generate_annotated_pipelines --skip_node_sampling --skip_train_policies --skip_generate_pipelines
```
