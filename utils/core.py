import itertools
import pathlib


def combinations(lst):
    return list(itertools.product(*lst))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def scan_folders(path: pathlib.Path):
    return list(filter(lambda f: f.is_dir(), path.glob("*")))
