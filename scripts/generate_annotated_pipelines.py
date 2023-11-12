import argparse

from scripts.steps import (
    annotate_pipelines,
    generate_pipelines_from_policies, 
    node_sampling, 
    train_policies_from_target_sets
)

parser = argparse.ArgumentParser(
    prog="Pipeline generator",
    description="Trains A3C policies based on generated target sets and generated pipelines using them",
)

parser.add_argument("--target_sets", type=int, default=200)
parser.add_argument("--policy_episodes", type=int, default=50)
parser.add_argument("--sampling_rate", type=float, default=0.5)
parser.add_argument("--min_item_set_nodes", type=int, default=100)
parser.add_argument("--max_item_set_nodes", type=int, default=10000)
parser.add_argument("--min_pipeline_size", type=int, default=4)
parser.add_argument("--max_pipeline_size", type=int, default=8)

args = parser.parse_args()

node_sampling(
    args.target_sets,
    args.sampling_rate, 
    args.min_item_set_nodes,
    args.max_item_set_nodes
)

train_policies_from_target_sets(args.policy_episodes)

generate_pipelines_from_policies(
    min_pipeline_size=args.min_pipeline_size, 
    max_pipeline_size=args.max_pipeline_size
)

annotate_pipelines()
