#!/usr/bin/env python3
# Created on Sun Mar 16 09:08:06 2025
# author: paul
# Coppied from https://docs.pytest.org/en/latest/example/simple.html#incremental-testing-test-steps
import numpy as np

from pathlib import Path
import zipfile

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

@pytest.fixture(params=['thresh', 'mean', 'max'])
def irfstyle(request):
    return request.param


@pytest.fixture
def data()->smf.PhotonData:
    raw = smf.hdf5.load('data/HP3_TE300_SPC630.hdf5')
    data = smf.hdf5.regularize_dets(raw)
    bg = smf.ff.make_bg(data.detdef, auto_threshold=True)
    burst = smf.Param(smf.Bursts, m=10, F=6.0, stream=smf.PhSel('0ex_1em'), bg=bg['bg'])
    nph = smf.Column(burst, 'nph_raw', smf.PhSel('0ex_1em'))
    gate = smf.make_geq_gate(nph, 70)
    burst = burst.regate(gate)
    bs_irfex = smf.Param(smf.Bursts, m=10, F=1.01, stream=smf.PhSel('0ex_1em'), bg=bg['bg'])
    irf_p = smf.Param(smf.BurstOvlp, truthtable='inv', bases=bs_irfex)
    for sel in (smf.PhSel('0ex0em'), smf.PhSel('0ex1em'), smf.PhSel('1ex1em')):
        data.irf_thresh[sel] = np.argmax(data.get_column(smf.Column(burst, 'nanohist', (sel, True))).sum(axis=0))
        irf = data.get_column(smf.Column(irf_p, 'nanohist', sel)).sum(axis=0)
        irf[irf < 0.5*irf.max()] = 0.0
        data.irf[sel] = irf
    return data


@pytest.fixture
def default_bg(data)->smf.Param:
    return smf.fretfactory.make_bg(data)['bg']


@pytest.fixture
def default_burst(default_bg):
    return smf.Param(smf.Bursts, bg=default_bg, m=10, F=6.0)


@pytest.fixture
def sper_bg(data)->smf.Param:
    return smf.fretfactory.make_bg(data, period=3600.0)['bg']


@pytest.fixture
def data1ex():
    raw = smf.hdf5.load('data/0023uLRpitc_NTP_20dT_0.5GndCl.hdf5')
    data = smf.hdf5.regularize_dets(raw)
    return data


@pytest.fixture
def datapolgroup():
    def process(file:str):
        if not Path(file).exists():
            with zipfile.ZipFile("data/Lab8_U2AF2.zip") as z:
                z.extractall('data/')
        raw = smf.lr.load_ptu(file)
        raw.setup['num_spectral_ch'] = 2
        raw.setup['num_polarization_ch'] = 2
        raw.setup['num_split_ch'] = 1
        raw.setup['excitation_wavelengths'] = np.array([532e-9, 642e-9])
        raw.setup['detection_wavelengths'] = np.array([585e-9, 698e-9])
        raw.setup['excitation_cw'] = np.array([False, False])
        raw.setup['excitation_alternated'] = np.array([False, False])
        raw.setup['detectors']['label'] = np.array(['ATTO 488', 'ATTO 643'])
    
        # remove the spectral_ploarization_split because it is there to identify detectors, but is not part of HDF5 spec
        raw.photon_data[0].meas_specs['detectors_specs'].pop('spectral_polarization_split_chN', None)
        # 
        raw.photon_data[0].meas_specs['detectors_specs']['spectral_ch1'] = np.array([2,3], dtype=np.uint8)
        raw.photon_data[0].meas_specs['detectors_specs']['spectral_ch2'] = np.array([4,5], dtype=np.uint8)
        raw.photon_data[0].meas_specs['detectors_specs']['polarization_ch1'] = np.array([2,4], dtype=np.uint8)
        raw.photon_data[0].meas_specs['detectors_specs']['polarization_ch2'] = np.array([3,5], dtype=np.uint8)
        raw.photon_data[0].meas_specs['alex_excitation_period1'] = np.array([1850,3000])
        raw.photon_data[0].meas_specs['alex_excitation_period2'] = np.array([70,1500])
        return smf.hdf5.regularize_dets(raw)
    
    def get_irf(d:smf.PhotonData):
        sels = ( d.detdef.stream_ids_to_PhSel(i) for i in range(d.detdef.size))
        irf_thresh = dict()
        irf = dict()
        for sel in sels:
            ex = list(sel.ex.elements)[0]
            start, stop = d.setup.ex_ranges[ex][0,:]
            nanohist = np.bincount(d.get_nanos(sel), minlength=stop)
            irf_thresh[sel] = np.argmax(nanohist)
            nanohist = nanohist[start:stop]
            nanohist -= np.mean(nanohist[:10], dtype=nanohist.dtype)
            nanohist[nanohist < nanohist.max()*0.1] = 0
            irf[sel] = nanohist
        return irf_thresh, irf
    
    files = ('data/Lab8_U2AF2/mystery_protein_2_holo_ulm_1.ptu', 
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_4.ptu', 
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_5.ptu',
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_6.ptu',
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_7.ptu',
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_8.ptu',
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_8.ptu',
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_9.ptu',
             'data/Lab8_U2AF2/mystery_protein_2_holo_ulm_10.ptu',
             )
    datas = [process(file) for file in files]
    data_irf = process('data/Lab8_U2AF2/BSA+buffer.ptu')
    irf_thresh, irf = get_irf(data_irf)
    for d in datas:
        d.irf = irf
        d.irf_thresh = irf_thresh
    return smf.PhotonDataList(datas)
