import torch


def pytest_sessionstart(session) -> None:
    torch.set_num_threads(1)
