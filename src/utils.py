import random


def train_test_split(
    examples,
    seed: int = 228,
    train_frac: float = 0.7,
):
    """чтоб не тащить весь sklearn в проект только ради этого"""
    assert train_frac < 1.0

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    num_train = int(len(examples) * train_frac)
    train = [examples[i] for i in indices[:num_train]]
    test = [examples[i] for i in indices[num_train:]]

    return train, test


def train_test_valid_split(
    examples,
    seed=228,
    train_frac=0.7,
    test_frac=0.2
):
    assert train_frac + test_frac < 1.0

    train, test_valid = train_test_split(examples, seed=seed, train_frac=train_frac)
    test_frac = test_frac / (1.0 - train_frac)
    test, valid = train_test_split(test_valid, seed=seed, train_frac=test_frac)

    return train, valid, test
