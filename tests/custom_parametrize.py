import pytest


def parametrize_from_dict(test_suite: dict[str, dict]) -> pytest.MarkDecorator:
    args = list(next(iter(test_suite.values())).keys())
    values = [[item[a] for a in args] for item in test_suite.values()]
    ids = list(test_suite.keys())
    return pytest.mark.parametrize(args, values, ids=ids)
