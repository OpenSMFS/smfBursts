# -*- coding: utf-8 -*-
# Author : Paul David Harris
# Created 06/12/2025
"""
Module for loading raw data from PhotonHDF5 files, and creating processed
:class:`photondata.PhotonData` objects.
"""

import re
import weakref
from warnings import warn
from os import PathLike
from typing import Union, Any, ClassVar
from collections.abc import Callable

import numpy as np
import tables as tb

from phconvert.hdf5 import Invalid_PhotonHDF5, dict_from_group, save_photon_hdf5
from phconvert.helperfuncs import pop_nones

from .datamodel.utils import (_DataLike, MutDict, tupledict, make_objectarray, 
                         enumerate_intersects, _FileFinalizer, ImDict, _eq, 
                         _GroupFuture, GroupFuture)
from .datamodel.diskdict import MappedAttrDD, TypedValueDD
from .photondata import PhSpec, PhotonData, PhotonDataList, TV_pharray_mtch, PhArray, regularize_photon_data
from .ph_sel import DetDef, PhSel


class PhEventsRawDiskDict(MappedAttrDD, TypedValueDD):
    """
    Disk dictionary to represent specifically raw photon arrays of photondata.
    This is always intended to be an attribute of :class:`PhGroupRaw`, as
    it does not store any of the settings.
    
    There are 4 keys:
    #. times: photon arrival times
    #. dets: detector index of each photon
    #. nanos: photon nanotimes (pulsed data only)
    #. particles : simulated particle index (simulated data only)
    """
    _name_map = ImDict({'times':'timestamps', 'dets':'detectors', 
                       'nanos':'nanotimes', 'particles':'particles'})
    _typemap = ImDict(times=TV_pharray_mtch(dtype=np.int64), dets=TV_pharray_mtch(dtype=np.uint8),
                      nanos=TV_pharray_mtch(dtype=np.uint16), particles=TV_pharray_mtch(dtype=np.uint16))
    _exclude_groups = ('user', 'measurement_specs', 'timestamps_specs', 'nanotimes_specs')
    _mut:bool # if 
    
    def __init__(self, dct:Union[dict,None]=None, group:Union[tb.Group,None]=None):
        self._in_init = True
        super().__init__(dct, group)
        delattr(self, '_in_init')
        self._mut = False
    
    @classmethod
    def _valtype(cls, key):
        """Type of array for key"""
        return cls._typemap[key]
    
    def _read_group(self, group:tb.Group, nodename:str)->Any:
        """read specific node from gruop"""
        return group[nodename].read()
    
    def _write_group(self, group:tb.Group, nodename:str, value:Any)->tb.Group:
        """Write specific node to group"""
        group._v_file.create_carray(group, nodename, value)
    
    def __getattr__(self, attr):
        if attr not in self._name_map:
            raise AttributeError(f"{type(self).__name__} has no attribute {attr}")
        if attr in self:
            return self[attr]
        raise AttributeError(f"{type(self).__name__} does not have a {attr} array")
    
    def __setitem__(self, name, val):
        if hasattr(self, '_in_init') and self._in_init:
            super().__setitem__(name, val)
            return
        if name in val and _eq(val, self[name]):
            return
        self.reset_group()
        super().__setitem__(name, val)
        self._mut = True
    
    def __setattr__(self, name, val):
        if name in self._name_map:
            self.__setitem__(name, val)
        else:
            super().__setattr__(name, val)

    @property
    def mut(self)->bool:
        """If any data was modified after creation"""
        return self._mut
    
    @property
    def as_photonHDF5_dict(self)->dict:
        return {self._name_map[k]:v for k, v in self.items()}


class PhGroupRaw(_DataLike):
    """
    Stores single photon_dataX group information
    
    Parameters
    ----------
    photon_data : PhEventsRawDiskDict
        raw photon data (times, dets, nanos, particles)
    meas_specs : MutDict
        dictionary of measurement specs.
    time_specs : MutDict
        dictionary of timespecs (must specify 'timestamps_unit')
    nano_specs : MutDict
        dictionary of nanotimes specs (if present, must specify 'tcspc_unit')
        
    """
    __slots__ = ('photon_data', 'meas_specs', 'time_specs', 'nano_specs')
    _name_map = tupledict(('time_specs', 'timestamps_specs'), 
                          ('meas_specs', 'measurement_specs'), 
                          ('nano_specs', 'nanotimes_specs'))
    _ex_def = (re.compile(r'^alex_excitation_period(\d*[1-9])$'), 1)
    _em_def = (re.compile(r'^spectral_ch(\d*[1-9])$'), 0)
    _pol_def = (re.compile(r'^polarization_ch(\d*[1-9])$'), 0)
    _split_def = (re.compile(r'^split_ch(\d*[1-9])$'), 0)
    _non_photon_def = (re.compile(r'^non_photon_id(\d*[1-9])$'), 0)

    photon_data: PhEventsRawDiskDict
    meas_specs : MutDict
    time_specs : MutDict
    nano_specs : MutDict

    def __post_init__(self):
        if not isinstance(self.photon_data, PhEventsRawDiskDict):
            raise TypeError("photon_data must be PhEventsRawDiskDict")
        if not isinstance(self.meas_specs, MutDict):
            raise TypeError("meas_specs must be MutDict")
        if not isinstance(self.time_specs, MutDict):
            raise TypeError("time_specs must be MutDict")
        if 'nano_specs' in self and not isinstance(self.nano_specs, MutDict):
            raise TypeError("nano_specs must be MutDict")

    @classmethod
    def load_group(cls, phdata:tb.Group, ondisk:bool=False)->"PhGroupRaw":
        """
        Load a photon_dataX gruop from HDF5 group as PhGroupRaw

        Parameters
        ----------
        phdata : tb.Group
            photon_dataX group.
        ondisk : bool, optional
            If to keep on disk (True) or load into memory (False). The default is False.

        Returns
        -------
        PhGroupRaw
            In memory representation of photon_data group.

        """
        kwargs = dict(
            # photon_data = PhEventsRawDiskDict(group=phdata, ondisk=ondisk),
            photon_data = PhEventsRawDiskDict(group=phdata),
            meas_specs = MutDict(dict_from_group(phdata.measurement_specs)),
            time_specs = MutDict(dict_from_group(phdata.timestamps_specs))
                      )
        if 'nanotimes_specs' in phdata:
            kwargs['nano_specs'] = MutDict(dict_from_group(phdata.nanotimes_specs))
        if not ondisk:
            kwargs['photon_data'].reset_group()
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, phdata:dict)->"PhGroupRaw":
        """
        Create a new instance from a dict with photon-HDF5 like format

        Parameters
        ----------
        phdata : dict
            Input data to create new object from.

        Returns
        -------
        PhGroupRaw
            Data represented by dict.

        """
        inv = {v:k for k, v in PhEventsRawDiskDict._name_map.items()}
        allowed = tuple(cls._name_map.values()) + tuple(inv.keys())
        if any((err:=k) not in allowed for k in phdata.keys()):
            raise KeyError(f'{err} is not a valid key for photon_data group')
        kwargs = dict(
            photon_data=PhEventsRawDiskDict({inv[k]:v for k, v in phdata.items() if k in inv}),
            meas_specs = MutDict(phdata['measurement_specs']),
            time_specs = MutDict(phdata['timestamps_specs'])
                     )
        if 'nanotimes_specs' in phdata:
            kwargs['nano_specs'] = MutDict(phdata['nanotimes_specs'])
        return cls(**kwargs)

    @property
    def spec_mut(self)->bool:
        """Have any of the specs changed from creation"""
        return any([self.meas_specs.mut, self.time_specs.mut, self.nano_specs.mut])

    def _get_channel_sequence(self, spec:dict[str,np.ndarray[np.number]], rgx:re.Pattern,
                              max_skip:int, n:Union[int,None]=None)->np.ndarray[np.object_]:
        """For given spec and channel (defined by rgx) get each [channel]X in order in spec"""
        seq = list()
        for k, v in spec.items():
            mtch = rgx.match(k)
            if not mtch:
                continue
            i = int(mtch.group(1))
            while len(seq) < i:
                seq.append(None)
            seq[i-1] = v
        if n is not None:
            seq += [None for _ in range(len(seq), n)]
        if n is not None and len(seq) < n:
            raise ValueError("more values than specified")
        if sum([s is None for s in seq]) > max_skip:
            raise ValueError('too many skiped channel definitions')
        return seq

    @property
    def times(self)->np.ndarray[np.int64]:
        """Raw photon arrival times, for single photon_dataX"""
        return self.photon_data.times

    @property
    def dets(self)->np.ndarray[np.uint8]:
        """Raw (un-sorted) photon detector indexes, for single photon_dataX"""
        return self.photon_data.dets

    @property
    def nanos(self)->np.ndarray[np.uint16]:
        """Raw photon nanotimes, for single photon_dataX"""
        return self.photon_data.nanos

    @property
    def particles(self)->np.ndarray[np.integer]:
        """Raw particles (simulated data only), for single photon_dataX"""
        return self.photon_data.particles

    @property
    def nex(self)->int:
        """number of excitation channels"""
        return len(self._get_channel_sequence(self.meas_specs, *self._ex_def))

    def exs(self, n=None)->np.ndarray[np.ndarray[np.int64],...]:
        """array of excitation periods"""
        return make_objectarray([np.asarray(ex).reshape(-1,2) for ex in
                                 self._get_channel_sequence(self.meas_specs,
                                                            *self._ex_def, n=n)])
    
    @property
    def nem(self)->int:
        """number of emission channels"""
        return len(self._get_channel_sequence(self.meas_specs['detectors_specs'], *self._em_def))

    def ems(self, n=None)->np.ndarray[np.ndarray[np.float64],...]:
        """detector indexes of each emission channel"""
        return make_objectarray(self._get_channel_sequence(self.meas_specs['detectors_specs'], *self._em_def, n=n))

    @property
    def npol(self)->int:
        """Number of polarization channels"""
        return len(self._get_channel_sequence(self.meas_specs['detectors_specs'], *self._pol_def))

    def pols(self, n=None)->np.ndarray[np.ndarray[np.float64],...]:
        """detector indexes of each polarization channel"""
        return make_objectarray(self._get_channel_sequence(self.meas_specs['detectors_specs'], *self._pol_def, n=n))

    @property
    def nsplit(self)->int:
        """number of split channels"""
        return len(self._get_channel_sequence(self.meas_specs['detectors_specs'], *self._split_def))

    def splits(self, n=None)->np.ndarray[np.ndarray[np.float64],...]:
        """detector indexes of each split channel"""
        return make_objectarray(self._get_channel_sequence(self.meas_specs['detectors_specs'], *self._split_def, n=n))

    def non_photons(self)->tuple[np.ndarray]:
        """detector indexes not representing photons"""
        return make_objectarray(self._get_channel_sequence(self.meas_specs, *self._non_photon_def))
    
    @property
    def as_photonHDF5_dict(self)->dict:
        """photon-HDF5 formatted dict of photon_dataX group"""
        out = dict()
        for key, val in self.items():
            if key == 'photon_data':
                out.update(val.as_photonHDF5_dict)
            else:
                out[self._name_map[key]] = val
        return out
    
    @property
    def mut(self)->bool:
        """If data or settigns were modified after creation."""
        return any(val.mut for val in self.values())
    
    @property
    def mut_ph(self)->bool:
        """If specifically photon data were modified after creation"""
        return self.photon_data.mut
    
    @property
    def mut_set(self)->bool:
        """If specifically settings were modified after creation"""
        return any(val.mut for key, val in self.items() if key != 'photon_data')


class SetupSpec(MutDict):
    """
    Special subclass of MutDict (so that user changes can be detected) that
    represents ``Setup`` group of photon-HDF5.
    """
    @classmethod
    def load_group(cls, group:tb.Group)->"SetupSpec":
        return cls(dict_from_group(group))


class PhotonHDF5Data(_DataLike):
    """
    Full representation of photon-HDF5 data. use :meth:`PhotonHDF5Data.load_hdf5`
    to load from file.
    """
    __slots__ = ('photon_data', 'setup', 'acquisition_duration', 'description',
                 'sample', 'identity', 'provenance', 'filename', 'user',
                 'ondisk', '_file', '_track', '_finalizer')
    _hdf5_exclude:ClassVar[frozenset[str]] = frozenset({'filename', 'ondisk', '_file', '_track', '_finalizer'})
    _photon_data_regex:ClassVar[re.Pattern] = re.compile(r'photon_data(?P<num>\d+)?')
    
    photon_data: tuple[PhGroupRaw,...]
    setup: SetupSpec
    description:str
    sample:dict[str,Union[int,str]]
    identity:dict[str,str]
    provenance:dict[str,str]
    acquisition_duration:str
    user:dict[str,Any]
    _file : Union[None, tb.File]
    _track: bool
    _finalizer:Callable

    def __post_init__(self):
        if 'photon_data' not in self:
            raise ValueError("must specify photon_data")
        if 'setup' not in self:
            raise ValueError("must specify setup")
        if not isinstance(self.photon_data, tuple):
            try:
                self.photon_data = tuple(self.photon_data)
            except Exception as e:
                raise TypeError('photon data must be tuple of PhGroupRaw') from e
        if any(not isinstance(pd, PhGroupRaw) for pd in self.photon_data):
            raise TypeError("photon_data must be tuple of PhGroupRaw")
        if not isinstance(self.setup, SetupSpec):
            raise TypeError("setup must be SetupSpec")
        if '_file' not in self:
            self._file = None
        if '_track' not in self:
            self._track = not self._file is None
        if isinstance(self._file, tb.File) and 'filename' not in self:
            self.filename = self.file.filename
        if isinstance(self._file, tb.File) and self._track:
            self._finalizer = _FileFinalizer(self._file, self)
        else:
            self._finalizer = None
        if 'ondisk' not in self:
            self.ondisk = self._file is not None and self._file.isopen
            
    def _finalize(self):
        """Release file from tracking (will close if file is not tracked by other objects)"""
        self._finalizer.finalize_owner(weakref.ref(self))

    @classmethod
    def _load_hdf5(cls, file:tb.File, ondisk:bool=False, track:bool=True, strict:bool=True)->"PhotonHDF5Data":
        """Internal loader of HDF5 file, once settings have been determined"""
        root = file.root
        photon_data = {cls._photon_data_regex.fullmatch(g._v_name).group('num'):g
                       for g in root._f_iter_nodes() if cls._photon_data_regex.fullmatch(g._v_name)}
        if None in photon_data and len(photon_data) != 1:
            raise Invalid_PhotonHDF5("photon_data and photon_dataX groups specified")
        photon_data:tuple[str,...] = tuple(photon_data[k] for k in sorted(photon_data.keys()))
        if photon_data is None:
            raise Invalid_PhotonHDF5("Missing expected photon_data groups")
        photon_data = tuple(PhGroupRaw.load_group(pharray, ondisk=ondisk) for pharray in photon_data)
        setup = SetupSpec.load_group(root.setup)
        if setup['num_spots'] != len(photon_data):
            if strict:
                raise Invalid_PhotonHDF5("Inconsistent number of spots")
            warn("Invalid PhotonHDF5 file, differing number of spots and photon_data groups")
        return cls(photon_data=photon_data, setup=setup, _file=file, _track=track)

    @classmethod
    def load_hdf5(cls, filename:Union[str,PathLike], ondisk:bool=False, track:bool=True)->"PhotonHDF5Data":
        """
        Load a photon-HDF5 file.

        Parameters
        ----------
        filename : str | os.PathLike
            Path-like object or string representing the file to load.
        ondisk : bool, optional
            Whether, when converting, to store new data onto disk, and keep
            hdf5-file open. The default is False.

        Raises
        ------
        TypeError
            bad filename.

        Returns
        -------
        PhotonHDF5Data
            python representation of photonHDF5 data.

        """
        if isinstance(filename, tb.File):
            file = filename
            close = False
        else:
            file = tb.open_file(filename, 'a' if ondisk else 'r')
            close = not ondisk
        try:
            out = cls._load_hdf5(file, ondisk=ondisk, track=track)
        except Exception as e:
            close = True
            raise e
        finally:
            if close:
                file.close()
        out.ondisk = ondisk
        return out

    @classmethod
    def from_dict(cls, data:dict)->"PhotonHDF5Data":
        """
        Create a new instance of :class:`PhotonHDF5Data` based on a dictionary
        of values formated like a photon HDF5 file 
        (like those from ``phconvert.loader.loadfile_...()`` functions).

        Parameters
        ----------
        data : dict
            Dictionary of data in a photon-HDF5 like fomrat, like those resulting
            from ``phconvert.loader.loadfile_...()`` functions.

        Raises
        ------
        ValueError
            Dictionary contains invalid fields for photon HDF5.

        Returns
        -------
        PhotonHDF5Data
            :class:`PhotonHDF5Data` object representing dictionary.

        """
        phkeys = sorted(k for k in data.keys() if cls._photon_data_regex.fullmatch(k))
        if len(phkeys) == 0:
            raise ValueError("No photon_data groups in dict, cannot read as photon HDF5 dict")
        if "photon_data" in phkeys and len(phkeys) != 1:
            raise ValueError("Invalid photon HDF5 dict, has photon_data group and photon_data0... gruops")
        kwargs = {k:v for k, v in data.items() 
                  if k != 'setup' and not cls._photon_data_regex.fullmatch(k)}
        
        kwargs.update(setup=SetupSpec(data['setup']))
        kwargs.update(photon_data=tuple(PhGroupRaw.from_dict(data[k]) for k in phkeys))
        kwargs['filename'] = kwargs.pop('_filename')
        return cls(**kwargs)
    
    @property
    def file(self)->Union[None,tb.File]:
        """File where raw photonHDF5 data is stored"""
        return self._file
    
    @file.setter
    def file(self, file:tb.File):
        if not isinstance(file, tb.File):
            raise TypeError("file must be  tables.File object")
        if self._track:
            self._finalizer = _FileFinalizer(file, self)

    @property
    def n_ch(self)->int:
        """number of spots (photon_dataX)"""
        return len(self.photon_data)
    # set of property methods to access specific attributes
    @property
    def times(self)->np.ndarray[np.ndarray[np.int64],...]:
        """Raw photon arrival times (macrotimes), in each spot"""
        return make_objectarray([pd.photon_data.times for pd in self.photon_data])

    @property
    def dets(self)->np.ndarray[np.ndarray[np.uint8],...]:
        """Raw photon detector indexes, in each spot"""
        return make_objectarray([pd.photon_data.dets for pd in self.photon_data])

    @property
    def nanos(self)->np.ndarray[np.ndarray[np.uint16],...]:
        """Raw photon nanotimes, in each spot"""
        return make_objectarray([pd.photon_data.nanos for pd in self.photon_data])

    @property
    def particles(self)->np.ndarray[np.ndarray[np.integer],...]:
        """Raw particle indexes (simulated data only), in each spot"""
        return make_objectarray([pd.photon_data.particles for pd in self.photon_data])
    
    @property
    def mut(self)->bool:
        """If any values have been changed after loading"""
        return self.setup.mut or any(pd.mut for pd in self.photon_data)

    def get_ex(self)->int:
        """Return number of excitation windows"""
        if any(self.photon_data[0].nex != ph_data.nex for ph_data in self.photon_data[1:]):
            raise ValueError("Inconsistent numbers of excitation lasers in each spot")
        return self.photon_data[0].nex if self.photon_data[0].nex != 0 else 1

    def get_em(self)->int:
        """return number of emission channels"""
        if any(self.photon_data[0].nem != ph_data.nem for ph_data in self.photon_data):
            raise ValueError("Inconsistent numbers of spectral channels in each spot")
        return self.photon_data[0].nem if self.photon_data[0].nem != 0 else 1

    def get_pol(self)->int:
        """return number of polarization channels"""
        if any(self.photon_data[0].npol != ph_data.npol for ph_data in self.photon_data):
            raise ValueError("Inconsistent numbers of spectral channels in each spot")
        return self.photon_data[0].npol if self.photon_data[0].npol != 0 else 1

    def get_split(self)->int:
        """return number of split channels"""
        if any(self.photon_data[0].nsplit != ph_data.nsplit for ph_data in self.photon_data):
            raise ValueError("Inconsistent numbers of spectral channels in each spot")
        return self.photon_data[0].nsplit if self.photon_data[0].nsplit != 0 else 1

    def get_offset_nanos(self, channel:int)->np.ndarray[np.uint16]:
        """
        Get the nanotimes (as numpy array) with the tcspc_offset applied

        Parameters
        ----------
        channel : int
            Which photon_dataX object to analyze.

        Returns
        -------
        np.ndarray[np.uint16]
            Nanotimes with offset applied.

        """
        if 'detectors' not in self.setup or 'tcspc_offset' not in self.setup['detectors']:
            return self.photon_data[channel].nanos
        ids = self.setup['detectors']['id']
        mask = self.setup['detectors']['spot'] == channel if len(self.photon_data) > 1 else np.ones(ids.shape, dtype=np.bool_)
        offsets = self.setup['detectors']['tcspc_offset']
        ids = ids[mask]
        offsets = offsets[mask]
        if np.all(offsets == 0):
            return self.photon_data[channel].nanos
        dets, nanos = self.photon_data[channel].dets, self.photon_data[channel].nanos[:].copy()
        for i, offset in zip(ids, offsets):
            nanos[dets == i] -= offset
        return nanos
    
    def _save_group_future(self)->_GroupFuture:
        """Get group future for self- avoids creation if not necessary"""
        file = self._file
        if file is None or not file.isopen:
            return _GroupFuture(None)
        if 'user/FRETBursts' in file.root:
            return _GroupFuture(self._file.root['user/FRETBursts'], self)
        def create_group()->tb.Group:
            if 'user' not in self._file.root:
                file.create_group(self._file.root, 'user')
            if 'FRETBursts' not in file.root.user:
                file.create_group(file.root.user, 'FRETBursts')
            return file.root.user.FRETBursts
        return _GroupFuture(create_group, self._finalizer, file=file)
    
    def _save_photon_group_future(self, i:None|int, groupfuture:_GroupFuture=None)->_GroupFuture:
        """Get/create GroupFuture for specific photon_data group"""
        if i is not None and (not isinstance(i, int) or i < 0):
            raise TypeError("i must be positive in or None")
        if groupfuture is None:
            groupfuture = self._save_group_future()
        groupname = 'photon_data' if i is None else f'photon_data{i}'
        return _GroupFuture.create_dependant(groupfuture._create_groupfuture(groupname), groupfuture, file=self._file)
    
    @property
    def as_photonHDF5_dict(self)->dict:
        """
        Photon-HDF5 like dict of data. Can be passed to ``phconvert.hdf5.save_photon_hdf5()``.
        """
        out = {k:v for k, v in self.items() 
               if not self._photon_data_regex.fullmatch(k) and k not in self._hdf5_exclude}
        if len(self.photon_data) == 1:
            out.update({'photon_data':self.photon_data[0].as_photonHDF5_dict})
        else:
            out.update({f'photon_data{i}':v.as_photonHDF5_dict for i, v in enumerate(self.photon_data)})
        return out
    
    def save_photonHDF5(self, file:str|PathLike|tb.File=None, close:bool=None, **kwargs)->tb.File:
        """
        

        Parameters
        ----------
        file : str|PathLike|tb.File, optional
            Path to file, or file into which to save data. The default is None.
        close : bool, optional
            Whether to close the created tb.File object at end of function call. 
            If None, close only if specied as str or PathLike, but if specified
            as tb.File, do not close.
            The default is None.
        **kwargs : Any
            Additional kwargs passed to phconvert.hdf5.save_photon_hdf5.

        Returns
        -------
        out : tb,File
            File object where data was saved.

        """
        if file is None:
            kwargs['h5_fname'] = '.'.join(file.filename.split('.')[:-1]) + 'hdf5'
        elif isinstance(file, tb.File):
            kwargs['h5file'] = file
        else:
            kwargs['h5_fname'] = file
        if close is None:
            close = not isinstance(file, tb.File)
        data = self.as_photonHDF5_dict
        pop_nones(data)
        out = save_photon_hdf5(data, close=close, **kwargs)
        if self.file is None:
            self.file = out
        return out


def _sort_tcspc_param(i:int, pd:PhGroupRaw, setup:SetupSpec, name:str,
                      em_dets:tuple[np.ndarray[np.uint8]],
                      pol_dets:tuple[np.ndarray[np.uint8]],
                      split_dets:tuple[np.ndarray[np.uint8]])->np.ndarray:
    """Extract sorted array tcspc param `name` from PhGroupRaw/PhotonHDF5Data"""
    size = np.prod([len(d) if len(d) != 0 else 1 for d in (em_dets, pol_dets, split_dets)])
    if name in setup['detectors']:
        rawarr = setup['detectors'][name]
        if 'spot' in setup['detectors']:
            rawarr = rawarr[setup['detectors']['spot'] == i]
        out = np.empty(size, dtype=np.float64)
        for shift, isect in enumerate_intersects(em_dets, pol_dets, split_dets):
            out[shift] = rawarr[isect[0]]
    else:
        out = np.repeat(pd.nano_specs[name], size)
    return out


# map of setup fields in HDF5 to name in python representation
_setup_field_map = (('excitation_wavelengths', 'ex_wv'), ('excitation_polarizations', 'ex_pol'),
                    ('excitation_input_powers', 'ex_pow'), ('excitation_intensity', 'ex_intensities'),
                    ('detection_wavelengths', 'em_wv_centers'), ('detection_polarizations', 'pol_angle'),
                    ('detection_split_ch_ratios', 'split_ratio'), ('alternated','alternated'))


def _get_uspecs(setup:SetupSpec)->dict:
    """Create dictionary of universal specs from PhotonHDF5Data"""
    return {outkey:setup[setupkey] for setupkey, outkey in _setup_field_map
            if setupkey in setup}


def _get_tcspc_param(i:int, pd:PhGroupRaw, setup:SetupSpec,
                     em_dets:tuple[np.ndarray[np.uint8]],
                     pol_dets:tuple[np.ndarray[np.uint8]],
                     split_dets:tuple[np.ndarray[np.uint8]])->dict:
    """Extract or infer all possible TCSPC parameters from PhGroupRaw and additional settings"""
    out = dict(tcspc_unit=_sort_tcspc_param(i, pd, setup, 'tcspc_unit', em_dets, pol_dets, split_dets))
    if 'tcspc_num_bins' in pd.nano_specs or 'tcspc_num_bins' in setup:
        out['tcspc_num_bins'] = _sort_tcspc_param(i, pd, setup, 'tcspc_num_bins', em_dets, pol_dets, split_dets)
    if 'tcspc_range' in pd.nano_specs:
        out['tcspc_range'] = pd.nano_specs['tcspc_range']
    return out


def load(filename:Union[str,PathLike], ondisk:bool=False):
    """
    Load a photon-HDF5 file.

    Parameters
    ----------
    filename : str | os.PathLike
        Path-like object or string representing the file to load.
    ondisk : bool, optional
        Whether, when converting, to store new data onto disk, and keep
        hdf5-file open. The default is False.

    Raises
    ------
    TypeError
        bad filename.
    
    Returns
    -------
    PhotonHDF5Data
        python representation of photonHDF5 data.

    """
    return PhotonHDF5Data.load_hdf5(filename, ondisk)


def regularize_dets(raw:PhotonHDF5Data, alex_type:str=None, autosave:bool=False, 
                    keepraw:Union[None,bool]=False, group:GroupFuture=None, 
                    track:bool=True, load_saved:bool=True, asdatalist:bool=True, 
                    unpack:bool=True)->Union[PhotonData,tuple[PhotonData,...],PhotonDataList]:
    """
    Create PhotonDataList object from PhotonHDF5Data. Assigns photons sorted dets
    indexes based on ex, em, pol, and split channels, and removes photons outside
    of excitation windows.

    Parameters
    ----------
    raw : PhotonHDF5Data
        Raw data to regularize.
    alex_type : str, optional
        One of ``'macro'``, ``'nano'``, ``'none'``. Specifies at which level
        laster alternation occurs, if ``None`` the value is automatially assigned
        based on `raw`. The default is None.
    autosave : bool, optional
        Autosave value for initial returned PhotonDataS object. The default is False.
    group : GroupFuture, optional
        HDF5 group where to store data. The default is None.
    load_saved : bool
        If :code:`True` then set group of returned data to user/FRETBursts/photon_dataX
        so that any previously saved columns can be loaded instead of calculated.
    track_file : bool, optional
        Whether or not the output object should be responsible for closing the file.
        The default is True
    asdatalist : bool, optional
        The default is True

    Returns
    -------
    PhotonDataList
        Procesed PhotonDataList object, all spots together.

    """
    if group is None:
        group = raw._save_group_future()
        file = raw._file
    else:
        if isinstance(group, (tb.Group, _GroupFuture)):
            file = group._v_file if isinstance(group, tb.Group) else group._filefuture
        else:
            file = None
        if not isinstance(group, _GroupFuture):
            group = _GroupFuture(group)
    pharrays = list()
    detdef = DetDef(ex=raw.get_ex(), em=raw.get_em(), pol=raw.get_pol(), split=raw.get_split())
    raw_setup = raw.setup
    uspecs = _get_uspecs(raw_setup)
    for i, pd in enumerate(raw.photon_data):
        em_dets = pd.ems()
        pol_dets = pd.pols()
        split_dets = pd.splits()
        sdict = dict(clk_p=pd.time_specs['timestamps_unit'], ex_ranges=pd.exs())
        sdict.update(uspecs)
        if hasattr(pd, 'nanos'):
            sdict.update(_get_tcspc_param(i, pd, raw_setup, em_dets, pol_dets, split_dets))
        if 'alex_period' in pd.meas_specs:
            sdict['alex_period'] = pd.meas_specs['alex_period']
        if 'alex_offset' in pd.meas_specs:
            sdict['alex_offset'] = pd.meas_specs['alex_offset']
        if alex_type is None:
            if 'alex_period' in sdict:
                sdict['alex_type'] = 'macro'
            elif 'tcspc_unit' in sdict or 'tcspc_range' in sdict or 'tcspc_num_bins' in sdict:
                sdict['alex_type'] = 'nano'
            else:
                sdict['alex_type'] = 'none'
        setup = PhSpec(detdef=detdef, **sdict)
        skwargs = dict(em_dets=em_dets, pol_dets=pol_dets, split_dets=split_dets)
        if hasattr(pd, 'nanos'):
            skwargs['nanos'] = pd.nanos
        if hasattr(pd, 'particles'):
            skwargs['particles'] = pd.nanos
        pharrays.append(regularize_photon_data(setup, pd.times, pd.dets, **skwargs))
    if keepraw is None:
        keepraw = raw.file is None or raw.ondisk
    ref = raw if keepraw else weakref.ref(raw)
    save_group_future = raw._save_group_future() if load_saved else None
    out = tuple(PhotonData(pharray, group=save_group_future, group_no=i,
                           meta={'filename':raw.filename}, autosave=autosave, 
                           file=file, track=track, ref=ref)
                for i, pharray in enumerate(pharrays))
    if unpack and len(out) == 1:
        out = out[0]
    elif asdatalist:
        out = PhotonDataList(out)
    return out