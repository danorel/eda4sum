# Train policies on sampled target sets

## Prerequisites

Requires python 3.8.15

Install required dependencies:

    python -m pip install -r requirements.txt

## Setup data 

Unzip data:

    cd app/data/
    cat data.tar.gz.* | tar xzvf -

Unzip models:

    cd app/app_models/
    cat app_models.tar.gz.* | tar xzvf -

## Run scripts 

1. Generate models policies

```shell
python -m scripts.generate_policies --policies 3 --episodes 1
```

2. Generate pipelines on trained policies 

```shell
python -m scripts.generate_pipelines
```
