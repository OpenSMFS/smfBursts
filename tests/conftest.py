#!/usr/bin/env python3
# Created on Sun Mar 16 09:08:06 2025
# author: paul
# Coppied from https://docs.pytest.org/en/latest/example/simple.html#incremental-testing-test-steps


import pytest

# store history of failures per test class name and per index in parametrize (if parametrize used)
_test_failed_incremental: dict[str: dict[tuple[int, ...]: str]] = dict()
_test_dependency_result: dict[str: bool] = dict()

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "incremental: mark test to run only on named environment"
    )
    config.addinivalue_line(
        "markers", "dependency: mark test to run only if depends passed"
    )


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        # incremental marker is used
        if call.excinfo is not None:
            # the test has failed
            # retrieve the class name of the test
            cls_name = str(item.cls)
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the test function
            test_name = item.originalname or item.name
            # store in _test_failed_incremental the original name of the failed test
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(
                parametrize_index, test_name
            )
    if call.when == 'call':
        if "dependency" in item.keywords:
            # dependency marker is used
            name = item.get_closest_marker('dependency').kwargs.get('name', None)
            if name is not None:
                if _test_dependency_result.get(name, None) is None:
                    _test_dependency_result[name] =  call.excinfo


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        # retrieve the class name of the test
        cls_name = str(item.cls)
        # check if a previous test has failed for this class
        if cls_name in _test_failed_incremental:
            # retrieve the index of the test (if parametrize is used in combination with incremental)
            parametrize_index = (
                tuple(item.callspec.indices.values())
                if hasattr(item, "callspec")
                else ()
            )
            # retrieve the name of the first test function to fail for this class name and index
            test_name = _test_failed_incremental[cls_name].get(parametrize_index, None)
            # if name found, test has failed for the combination of class name & test name
            if test_name is not None:
                pytest.skip(f"previous test failed ({test_name})")
    if 'dependency' in item.keywords:
        depends = item.get_closest_marker('dependency').kwargs.get('depends', None)
        if depends is not None:
            depends = (depends, ) if isinstance(depends, str) else depends
            # raise Exception(f"{depends}")
            if any(_test_dependency_result.get(dep, True) is not None for dep in depends):
                pytest.skip(f"previous test failed (one of {depends})")


import smfbursts as smf


@pytest.fixture(params=['start', 'istarttime'])
def colstart(request):
    return request.param


@pytest.fixture(params=['stop', 'istoptime'])
def colstop(request):
    return request.param

@pytest.fixture
def data()->smf.PhotonData:
    raw = smf.photonHDF5.load('data/HP3_TE300_SPC630.hdf5')
    data = smf.photonHDF5.regularize_dets(raw)
    return data

@pytest.fixture
def default_bg(data)->smf.Param:
    return smf.fretfactory.make_bg(data)['bg']

@pytest.fixture
def sper_bg(data)->smf.Param:
    return smf.fretfactory.make_bg(data, period=3600.0)['bg']