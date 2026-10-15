"""CPU diagnostics are opt-in and never count as the R1 GPU release gate."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--container-cuda", choices=("cu126", "cu128"), default="cu126",
                     help="locked CUDA wheel profile to build and test")
    parser.addoption("--container-device", choices=("cuda", "cpu"), default="cuda",
                     help="container inference device; cpu is diagnostic only, not the R1 GPU gate")


@pytest.fixture(scope="module")
def container_cuda(request):
    return request.config.getoption("--container-cuda")


@pytest.fixture(scope="module")
def container_device(request):
    return request.config.getoption("--container-device")
