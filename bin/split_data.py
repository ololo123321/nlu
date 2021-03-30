import os
import shutil
import random
from argparse import ArgumentParser


def rmkdir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def main(collection_dir, train_dir, valid_dir, test_dir, valid_frac=0.1, test_frac=0.2, seed=228):
    names = [x.split(".")[0] for x in os.listdir(collection_dir) if x.endswith(".txt")]
    n = len(names)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    valid_size = int(n * valid_frac)
    test_size = int(n * test_frac)
    train_size = n - valid_size - test_size

    train_names = [names[i] for i in indices[0:train_size]]
    valid_names = [names[i] for i in indices[train_size:train_size + valid_size]]
    test_names = [names[i] for i in indices[train_size + valid_size:]]

    def f(src_dir, dst_dir, names):
        rmkdir(dst_dir)
        for name in names:
            shutil.copy(os.path.join(src_dir, f"{name}.txt"), dst_dir)
            shutil.copy(os.path.join(src_dir, f"{name}.ann"), dst_dir)

    f(src_dir=collection_dir, dst_dir=train_dir, names=train_names)
    f(src_dir=collection_dir, dst_dir=valid_dir, names=valid_names)
    f(src_dir=collection_dir, dst_dir=test_dir, names=test_names)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--valid_frac", type=float, default=0.1, required=False)
    parser.add_argument("--test_frac", type=float, default=0.2, required=False)
    parser.add_argument("--seed", type=int, default=228, required=False)
    args = parser.parse_args()

    main(
        collection_dir=os.path.join(args.data_dir, "collection"),
        train_dir=os.path.join(args.data_dir, "train"),
        valid_dir=os.path.join(args.data_dir, "valid"),
        test_dir=os.path.join(args.data_dir, "test"),
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )
