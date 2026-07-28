#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for loading data from raw files
"""
from os import PathLike
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable, Hashable, Sequence
from itertools import chain
import re

import numpy as np

import phconvert as phc
from phconvert.loader import loadfile_ptu, loadfile_bh, loadfile_sm

from .photondata import PhotonData, PhotonDataList
from .photonHDF5 import PhotonHDF5Data, SetupSpec, PhGroupRaw, regularize_dets


def _none_pop(main:dict, **kwargs)->dict:
    """Remove None values from dictionary"""
    main = dict() if main is None else main
    keys = list(main.keys())
    for k in keys:
        if main[k] is None:
            main.pop(k)
    for k, v in kwargs.items():
        if v is None:
            continue
        main[k] = v
    return main


def _nonefill(dest:dict, name:str, main:dict, **kwargs):
    """If main is not None or kwargs are not None, add dictionary to dest under key name"""
    if main is None and not kwargs:
        return
    main = _none_pop(main, **kwargs)
    if name not in dest:
        dest[name] = dict()
    dest[name].update(main)


def _fill_hdf5_metadata(data:dict, ex_ranges, em_dets, pol_dets, split_dets, 
                        alex_period, alex_offset,
                        setup, sample, sample_name, dye_names, buffer_name,
                        identity, author, author_affiliation, provenance, )->dict:
    """Fill likely missing fields in """
    data.get('setup', dict()).get('measurement_specs', dict()).pop('spectral_polarization_split_chN', None)
    setupdct = SetupSpec(data['setup'])
    if setup is not None:
        setupdct.update(setup)
    data['setup'] = SetupSpec(data['setup'])
    _nonefill(data, 'sample', setup, dye_names=dye_names, buffer_name=buffer_name)
    if 'sample' in data and 'dye_names' in data['sample']:
        data['sample']['num_dyes'] = len(data['sample']['dye_names'].split(','))
    if 'description' in data and 'sample' in data and 'sample_name' not in data['sample']:
        data['sample']['sample_name'] = data['description']
    _nonefill(data, 'identity', identity, author=author, author_affiliation=author_affiliation)
    phc.helperfuncs.fill_measurement_type(data, 'generic')
    # phc.helperfuncs.pop_nones(data)
    raw = PhotonHDF5Data.from_dict(data)
    ex_dict = dict() if ex_ranges is None else _get_fill_fields(raw, 'alex_excitation_period', None, ex_ranges)
    ex_dct = _none_pop(ex_dict, alex_period=alex_period, alex_offset=alex_offset)
    if ex_dct:
        for phdata in raw.photon_data:
            phdata.meas_specs.update(ex_dct)
    if em_dets is not None:
        fill_emissions(raw, *em_dets)
    if pol_dets is not None:
        fill_polarizations(raw, *pol_dets)
    if split_dets is not None:
        fill_splits(raw, *split_dets)
    return raw


def load_ptu_phgroup(file:str|Path)->PhGroupRaw:
    """
    Load a ptu file as a :class:`smfbursts.photonHDF5.PhGroupRaw`

    Parameters
    ----------
    file : str|Path
        Path to PTU File.

    Returns
    -------
    PhGroupRaw
        PhGroupRaw data.

    """
    data = loadfile_ptu(file)
    return PhGroupRaw.from_dict(data['photon_data'])


def load_ptu(file:str|Path,
            ex_ranges:Sequence[np.ndarray[np.int64]]=None, 
            em_dets:Sequence[np.ndarray[np.uint8]]=None,
            pol_dets:Sequence[np.ndarray[np.uint8]]=None,
            split_dets:Sequence[np.ndarray[np.uint8]]=None,
            alex_period:int=None, alex_offset:int=None,
            setup:dict[str,str]=None, description:str=None, 
            sample:dict=None, sample_name:str=None, dye_names:str=None, buffer_name:str=None,
            identity:dict[str,str]=None, author:str=None, author_affiliation:str=None, 
            provenance:dict[str,str]=None)->PhotonHDF5Data:
    """
    Load .ptu (picoquant) file into a :class:PhotonHDF5Data` object.
    Note that excitation and emission windows will need to be completed to
    allow for regularization.

    Parameters
    ----------
    file : str | Path
        Path to ptu file.
    ex_ranges : Sequence[np.ndarray[np.int64]], optional
        The ranges of each excitation window for each laser. The default is None
    em_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each spectral (emission) channel. The default is None
    pol_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each polarization channel. The default is None
    split_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each split channel. The default is None
    alex_period : int, optional
        Alternation period (if known) in units of clocks. **usALEX only**.
        The default is None.
    alex_offset : int, optional
        Alternation period offset in units of clocks. **usALEX only**.
        The default is None.
    setup : dict[str,str], optional
        Fields to update ``/setup`` (according to photonHDF5 spec). 
        Note that some fields of setup are already filled based on metadata in ptu.
        The default is None.
    description : str, optional
        Description of experiment (in ``/description`` of photonHDF5 spec). 
        The default is None.
    sample : dict, optional
        ``/sample`` (according to photonHDF5 spec). The default is None.
    sample_name : str, optional
        ``/sample/sample_name`` field. The default is None.
    dye_names : str, optional
        ``/sample/dye_name`` field. The default is None.
    buffer_name : str, optional
       ``/samle/buffer_name`` field. The default is None.
    identity : dict[str,str], optional
        ``/identity`` (according to photonHDF5 spec). The default is None.
    author : str, optional
        ``/identity/author`` field. The default is None.
    author_affiliation : str, optional
        ``/identity/author_affiliation`` field. The default is None.
    provenance : dict[str,str], optional
        ``/provenance`` (according to photonHDF5 spec). The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data loaded from ptu file.

    """
    data, meta = loadfile_ptu(file)
    if description is None:
        if descr := meta['tags'].get('File_Comment', False):
            data['description'] = descr['data']
        else: 
            data['description'] = f'smfBursts imported from {file}'
    else:
        data['description'] = description
    return _fill_hdf5_metadata(data, ex_ranges, em_dets, pol_dets, split_dets, 
                               alex_period, alex_offset, setup, 
                               sample, sample_name, dye_names, buffer_name, 
                               identity, author, author_affiliation, provenance)


def load_spc_phgroup(file:str|Path, **kwargs)->PhGroupRaw:
    """
    Load a spc (Beckr & Hickl) file as a :class:`smfbursts.photonHDF5.PhGroupRaw`

    Parameters
    ----------
    file : str|Path
        Path to spc File.
    **kwargs
        Kwargs passed to phcovnert.loader.loadfile_bh

    Returns
    -------
    PhGroupRaw
        PhGroupRaw data.

    """
    data = loadfile_bh(file, **kwargs)
    return PhGroupRaw.from_dict(data['photon_data'])


def load_spc(file:str|Path, setfilename:str|Path=None, spc_model:str=None, SPC_type:str=None,
             ex_ranges:Sequence[np.ndarray[np.int64]]=None, 
             em_dets:Sequence[np.ndarray[np.uint8]]=None,
             pol_dets:Sequence[np.ndarray[np.uint8]]=None,
             split_dets:Sequence[np.ndarray[np.uint8]]=None,
             alex_period:int=None, alex_offset:int=None,
             setup:dict[str,str]=None, description:str=None, 
             sample:dict=None, sample_name:str=None, dye_names:str=None, buffer_name:str=None,
             identity:dict[str,str]=None, author:str=None, author_affiliation:str=None, 
             provenance:dict[str,str]=None)->PhotonHDF5Data:
    """
    Load .spc (Becker & Hickl) file into a :class:PhotonHDF5Data` object.
    Note that excitation and emission windows will need to be completed to
    allow for regularization.


    Parameters
    ----------
    file : str | Path
        name of file to load.
    setfilename : str | Path, optional
        Name of .set file, if not saem as filename. The default is None.
    spc_model : str, optional
        Model of SPC card, if None, auto-detect. Note that this is handed to 
        ``phconvert.loader.loadfile_bh``. The default is None.
    SPC_type : str, optional
        Type of SPC card, if None, auto-detect. Note that this is handed to 
        ``phconvert.loader.loadfile_bh``. The default is None.
    ex_ranges : Sequence[np.ndarray[np.int64]], optional
        The ranges of each excitation window for each laser. The default is None
    em_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each spectral (emission) channel. The default is None
    pol_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each polarization channel. The default is None
    split_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each split channel. The default is None
    alex_period : int, optional
        Alternation period (if known) in units of clocks.
        **usALEX only, rare for spc**. The default is None.
    alex_offset : int, optional
        Alternation period offset in units of clocks. 
        **usALEX only, rare for spc**. The default is None.
    setup : dict[str,str], optional
        Fields to update ``/setup`` (according to photonHDF5 spec). 
        Note that some fields of setup are already filled based on metadata in spc.
        The default is None.
    description : str, optional
        Description of experiment (in ``/description`` of photonHDF5 spec). 
        The default is None.
    sample : dict, optional
        ``/sample`` (according to photonHDF5 spec). The default is None.
    sample_name : str, optional
        ``/sample/sample_name`` field. The default is None.
    dye_names : str, optional
        ``/sample/dye_name`` field. The default is None.
    buffer_name : str, optional
       ``/samle/buffer_name`` field. The default is None.
    identity : dict[str,str], optional
        ``/identity`` (according to photonHDF5 spec). The default is None.
    author : str, optional
        ``/identity/author`` field. The default is None.
    author_affiliation : str, optional
        ``/identity/author_affiliation`` field. The default is None.
    provenance : dict[str,str], optional
        ``/provenance`` (according to photonHDF5 spec). The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data loaded from .spc file.

    """
    kw = {k:v for k, v in (('setfilename', setfilename), 
                           ('spc_model', spc_model), 
                           ('SPC_type', SPC_type)) if v is not None}
    data, meta = loadfile_bh(file, **kw)
    return _fill_hdf5_metadata(data, ex_ranges, em_dets, pol_dets, split_dets, 
                               alex_period, alex_offset, setup, 
                               sample, sample_name, dye_names, buffer_name, 
                               identity, author, author_affiliation, provenance)


def load_sm(file:str|Path,
            ex_ranges:Sequence[np.ndarray[np.int64]]=None, 
            em_dets:Sequence[np.ndarray[np.uint8]]=None,
            pol_dets:Sequence[np.ndarray[np.uint8]]=None,
            split_dets:Sequence[np.ndarray[np.uint8]]=None,
            alex_period:int=None, alex_offset:int=None,
            setup:dict[str,str]=None, description:str=None, 
            sample:dict=None, sample_name:str=None, dye_names:str=None, buffer_name:str=None,
            identity:dict[str,str]=None, author:str=None, author_affiliation:str=None, 
            provenance:dict[str,str]=None)->PhotonHDF5Data:
    """
    Load .sm file into a :class:PhotonHDF5Data` object.
    Note that excitation and emission windows will need to be completed to
    allow for regularization.


    Parameters
    ----------
    file : str | Path
        name of file to load.
    ex_ranges : Sequence[np.ndarray[np.int64]], optional
        The ranges of each excitation window for each laser. The default is None
    em_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each spectral (emission) channel. The default is None
    pol_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each polarization channel. The default is None
    split_dets : Sequence[np.ndarray[np.uint8]], optional
        Detector indexes for each split channel. The default is None
    alex_period : int, optional
        Alternation period (if known) in units of clocks. 
        The default is None.
    alex_offset : int, optional
        Alternation period offset in units of clocks. The default is None.
    setup : dict[str,str], optional
        Fields to update ``/setup`` (according to photonHDF5 spec). 
        Note that some fields of setup are already filled based on metadata in .sm.
        The default is None.
    description : str, optional
        Description of experiment (in ``/description`` of photonHDF5 spec). 
        The default is None.
    sample : dict, optional
        ``/sample`` (according to photonHDF5 spec). The default is None.
    sample_name : str, optional
        ``/sample/sample_name`` field. The default is None.
    dye_names : str, optional
        ``/sample/dye_name`` field. The default is None.
    buffer_name : str, optional
       ``/samle/buffer_name`` field. The default is None.
    identity : dict[str,str], optional
        ``/identity`` (according to photonHDF5 spec). The default is None.
    author : str, optional
        ``/identity/author`` field. The default is None.
    author_affiliation : str, optional
        ``/identity/author_affiliation`` field. The default is None.
    provenance : dict[str,str], optional
        ``/provenance`` (according to photonHDF5 spec). The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data loaded from .sm file.

    """
    data = loadfile_sm(file)
    return _fill_hdf5_metadata(data, ex_ranges, em_dets, pol_dets, split_dets, 
                               alex_period, alex_offset, setup, 
                               sample, sample_name, dye_names, buffer_name, 
                               identity, author, author_affiliation, provenance)


def get_detector_indexes(raw:PhotonHDF5Data, ich:None|int=None)->np.ndarray[np.uint8]:
    """
    Get unique indexes of detectors in raw data.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Data to extract indexes from.
    ich : None|int, optional
        Which channel to extract indexes from, if None, return union of all. 
        The default is None.

    Returns
    -------
    out : np.ndarray[np.uint8]
        Array of unique detector indexes.

    """
    if ich is None:
        unis = [np.unique(ph.dets) for ph in raw.photon_data]
    else:
        unis = [np.unique(raw.photon_data[0].dets)]
    out = unis[0]
    for uni in unis[1:]:
        out = np.union1d(out, uni)
    return out


def _get_fill_fields(raw:PhotonHDF5Data, field:str, ich:None|int, 
                     arrs:Sequence[np.ndarray])->tuple[range,dict[str,np.ndarray]]:
    """Return range of phdata indexes to update with fieldX arrays mapped from arrs"""
    ich = range(len(raw.photon_data)) if ich is None else range(ich, ich+1)
    fields = {f'{field}{i+1}':det for i, det in enumerate(arrs)}
    return ich, fields


def _fill_det_field(raw:PhotonHDF5Data, field:str, ich:int|None, det_arrs:Sequence[np.ndarray[np.uint8]])->PhotonHDF5Data:
    """Fill meas_specs['detector_specs'] with fieldX arrays"""
    ich, dets = _get_fill_fields(raw, field, ich, det_arrs)
    for i in ich:
        raw.photon_data[i].meas_specs['detector_specs'].update(dets)
    return raw


def fill_emissions(raw:PhotonHDF5Data, *args:np.ndarray[np.uint8], ich:None|int=None)->PhotonHDF5Data:
    """
    Fill out emission detectors specifications. Each arg is an array of all
    detectors in a given emission channel.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to update with emission metadata.
    *args : np.ndarray[np.uint8]
        Arrays of detector indexes for each excitation channel.
    ich : None|int, optional
        Which photon_dataX object to update, if None, update all with same information. 
        The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data updated. 
        (note that is same as input object, modification happen inplace).

    """
    return _fill_det_field(raw, 'spectral_ch', ich, args)


def fill_polarizations(raw:PhotonHDF5Data, *args:np.ndarray[np.uint8], ich:None|int=None)->PhotonHDF5Data:
    """
    Fill out polarization detectors specifications. Each arg is an array of all
    detectors in a given polarization channel.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to update with polarization metadata.
    *args : np.ndarray[np.uint8]
        Arrays of detector indexes for each polarization channel.
    ich : None|int, optional
        Which photon_dataX object to update, if None, update all with same information. 
        The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data updated. 
        (note that is same as input object, modification happen inplace).

    """
    return _fill_det_field(raw, 'polarization_ch', ich, args)


def fill_splits(raw:PhotonHDF5Data, *args:np.ndarray[np.uint8], ich:None|int=None)->PhotonHDF5Data:
    """
    Fill out split detectors specifications. Each arg is an array of all
    detectors in a given polarization channel.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to update with split metadata.
    *args : np.ndarray[np.uint8]
        Arrays of detector indexes for each split channel.
    ich : None|int, optional
        Which photon_dataX object to update, if None, update all with same information. 
        The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data updated. 
        (note that is same as input object, modification happen inplace).

    """
    return _fill_det_field(raw, 'split_ch', ich, args)


def fill_pie_windows(raw:PhotonHDF5Data, *args:np.ndarray[np.uint16], ich:None|int=None)->PhotonHDF5Data:
    """
    Fill out excitation window information for PIE (nsALEX) measurements.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to update with excitaiton window metadata.
    *args : np.ndarray[np.uint16]
        Arrays defining start/stop times of each excitation window.
    ich : None|int, optional
        Which photon_dataX object to update, if None, update all with same information. 
        The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data updated. 
        (note that is same as input object, modification happen inplace).

    """
    ich, exs = _get_fill_fields(raw, "alex_excitation_period", ich, args)
    for i in ich:
        raw.photon_data[i].meas_specs.update(exs)
    return raw


def fill_alex_windows(raw:PhotonHDF5Data, alternation_period:int, *args:np.ndarray[np.int64], offset:int=None, ich=None)->PhotonHDF5Data:
    """
    Fill out excitation window information for usALEX measurements.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to update with excitaiton window metadata.
    alternation_periods : int
        Duration (in mactrotime clocks) of alternation period.
    *args : np.ndarray[np.uint16]
        Arrays defining start/stop times of each excitation window.
    offset : int, optional
        Offset of excitation period in macrotime clocks. The default is None.
    ich : None|int, optional
        Which photon_dataX object to update, if None, update all with same information. 
        The default is None.

    Returns
    -------
    PhotonHDF5Data
        Raw data updated. 
        (note that is same as input object, modification happen inplace).

    """
    ich, exs = _get_fill_fields(raw, "alex_excitation_period", ich, args)
    for i in ich:
        raw.photon_data[i].meas_specs.update(exs)
        raw.photon_data[i].meas_specs['alternation_period'] = alternation_period
        if offset is not None:
            raw.photon_data[i].meas_specs['alex_offset'] = offset
    return raw


fill_nsalex_windows = fill_pie_windows
fill_usalex_windows = fill_alex_windows


def _regularize_finalizer(raw:list[PhotonHDF5Data], **kwargs:Any)->PhotonDataList:
    """Default finalizer function, takes sequence of PhotonHDF5Data and creates PhotonDataList"""
    return PhotonDataList([regularize_dets(r, **kwargs) for r in raw])


def _raw_finalizer(raw:list[PhotonHDF5Data], **kwargs:any)->PhotonHDF5Data:
    setup = raw[0].setup
    photon_data = tuple(chain.from_iterable(r.photon_data for r in raw))
    return PhotonHDF5Data(photon_data, setup, **kwargs)


_dir_finalizers = {None:_regularize_finalizer, True:_regularize_finalizer, 
                   'regularize':_regularize_finalizer, 
                   False:_raw_finalizer, 'raw':_raw_finalizer}


_pattern_dict = {'hdf5':(re.compile(r'.*\.hdf5'), PhotonHDF5Data.load_hdf5), 
                 'sm':(re.compile(r'.*\.sm'), load_sm), 
                 'ptu':(re.compile(r'.*\.ptu'), load_ptu),
                 'spc':(re.compile(r'.*\.spc'), load_spc) }


def load_dir(path:str|PathLike, pattern:re.Pattern|Literal['hdf5','sm','ptu','spc']='hdf5', 
             loader:Callable[[Path],PhotonHDF5Data]=None, 
             finalizer:Literal['regularize','raw']|Callable[[list[PhotonHDF5Data]],PhotonDataList]='regularize',
             loader_kwargs:dict[str:Any]=None, finalizer_kwargs:dict[str:Any]=None,
             sort:Callable[[str],Hashable]=None)->PhotonHDF5Data|PhotonDataList:
    """
    Load all files in directory path, matching pattern as a 
    :class:`smfbursts.photondata.PhotonDataList`.

    Parameters
    ----------
    path : str | PathLike
        Path to directory to load.
    pattern : re.Pattern|Literal['hdf5','sm','ptu','spc'], optional
        Pattern which files must match to be included. The default is 'hdf5'.
    loader : [[Path],PhotonHDF5Data], optional
        Load function, gets ``loader(filename, **loader_kwargs)``, should return
        a :class:`PhotonHDF5Data` object. The default is None.
    finalizer : {'regularize', 'raw'} | Callable, optional
        Finalization function, gets 
        ``finalizer(list[<result of loader>], **finalizer_kwargs)``, and should
        return a PhotonDataList object. If 'regularize'
        The default is 'regularize'.
    loader_kwargs : dict[str:Any], optional
        Kwargs passed to loader function. The default is None.
    finalizer_kwargs : dict[str:Any], optional
        Kwargs passed to finalizer. The default is None.
    sort : Callable[[str],Hashable], optional
        Function takes filename, and returns key to be use in sorting filenames.
        The default is None.

    Returns
    -------
    PhotonDataList
        :class:`smfbursts.photondata.PhotonDataList` object of all files in directory.

    """
    if isinstance(pattern, str):
        if pattern in _pattern_dict:
            pattern, ld = _pattern_dict[pattern]
            loader = ld if loader is None else loader
        else:
            pattern = re.compile(pattern)
    loader = PhotonHDF5Data.load_hdf5 if loader is None else loader
    loader_kwargs = dict() if loader_kwargs is None else loader_kwargs
    finalizer = finalizer if callable(finalizer) else _dir_finalizers.get(finalizer, finalizer)
    if not callable(finalizer):
        raise ValueError(f"Finalizer must be callable, 'regularize', or 'raw', not {finalizer}")
    finalizer_kwargs = dict() if finalizer_kwargs is None else finalizer_kwargs
    path = Path(path)
    files = sorted((f for f in path.iterdir() if pattern.match(f.name)), 
                   key = None if sort is None else lambda x: sort(x.name))
    return finalizer([loader(f, **loader_kwargs) for f in files], **finalizer_kwargs)