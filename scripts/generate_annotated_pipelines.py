import argparse

from scripts.steps import (
    annotate_pipelines,
    generate_pipelines_from_policies, 
    node_sampling, 
    train_policies_from_target_sets
)

parser = argparse.ArgumentParser(
    prog="Pipeline annotation job",
    description="Trains A3C policies based on generated target sets and generated pipelines using them",
)

parser.add_argument("--target_sets", type=int, default=200)
parser.add_argument("--policy_episodes", type=int, default=50)
parser.add_argument("--sampling_rate", type=float, default=0.5)
parser.add_argument("--min_item_set_nodes", type=int, default=100)
parser.add_argument("--max_item_set_nodes", type=int, default=10000)
parser.add_argument("--min_pipeline_size", type=int, default=4)
parser.add_argument("--max_pipeline_size", type=int, default=8)
parser.add_argument("--skip_node_sampling", default=False, action='store_true')
parser.add_argument("--skip_train_policies", default=False, action='store_true')
parser.add_argument("--skip_generate_pipelines", default=False, action='store_true')
parser.add_argument("--skip_annotate_pipelines", default=False, action='store_true')

args = parser.parse_args()

if not args.skip_node_sampling:
    node_sampling(
        args.target_sets,
        args.sampling_rate, 
        args.min_item_set_nodes,
        args.max_item_set_nodes
    )

if not args.skip_train_policies:
    train_policies_from_target_sets(args.policy_episodes)

if not args.skip_generate_pipelines:
    generate_pipelines_from_policies(
        min_pipeline_size=args.min_pipeline_size, 
        max_pipeline_size=args.max_pipeline_size
    )

if not args.skip_annotate_pipelines:
    annotate_pipelines()
