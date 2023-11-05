import argparse

from .utils.sampling.node_sampling import node_sampling
from .utils.generative import train_policies

parser = argparse.ArgumentParser(
                    prog='Policy Trainer',
                    description='Trains A3C policies based on generated target sets')

parser.add_argument('--policies', type=int, default=200)
parser.add_argument('--episodes', type=int, default=50)
parser.add_argument('--sampling_rate', type=float, default=0.5)
parser.add_argument('--min_item_set_nodes', type=int, default=100)
parser.add_argument('--max_item_set_nodes', type=int, default=10000)

args = parser.parse_args()

node_sampling(
    args.policies,
    args.sampling_rate,
    args.min_item_set_nodes,
    args.max_item_set_nodes
)
train_policies(args.episodes)
