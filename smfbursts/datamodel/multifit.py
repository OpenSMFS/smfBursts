#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Paul David Harris
# Created: 11 / 03 / 2026
# Purpose: histogramming of columns and fitting of multi-distributions
r"""
The multifit module is designed to make creating and optimizing distributions that
are composed of multiple distributions of the same type 
(e.g. multi-gaussian, multi-exponential) smoother.

It also includes some wrapper functions that make optimizing histogrammed values
easier.

``multi_...`` Functions
-----------------------

The ``multi_...``  functions are used to create multi-distribution functions from
a function for a single distribution. These all require specifying, in some way
the number of parameters per single distribution.
For instance a normal (gaussian) distribution has 2 parameters: sigma (standard devaiation)
and mu (center), while an exponential distribution is defined by a single paramater,
variously called decay constant, lambda, lifetime or tau depending on the formula
and whether the constant is placed in the numerator or denominator.

The ``multi_...`` funtions return a callable that has a signature of ``mdist(params:np.ndarray, x)``
where ``params`` is a 1D numpy array. ``x`` can vary, but it is typically either
a single 1D array or a sequence of such arrays, all of the same shape.
The params array follows the general form of 
``p0_0, ... a0, p1_0, ..., a1, ..., pN_M``  if the ``multi_...`` function does not end
in ``_free``, (note that the largest ``an`` param is ``a(n-1)``)
if it ends in ``_free``, then the form is ``p0_0, ... a0, p1_0, ..., a1, ..., pN_M, aM``
Where ``pn_m`` is the ``m``\th parameter of the ``m``\th distribution, and ``an`` 
is the amplitidue of the ``n``\th distribution. The returned multi distribtions
are of the form

.. math::
    
    F(\vec{x}) = \sum{a_{i}f(\vec{p})}

Amplitude rescaling
*******************

The non-free ``multi_...`` created params rescale the amplitudes so that each
input amplitude (:math:`a_{i}^{\prime}` ) can freely vary from 0 to 1 and the
realized amplitudes (:math:`a_{i}` ) *sum* to 1 (:math:`\sum^{N}{a_{i}} = 1` )

Rescaling is done by the :func:`retrieve_amps`.
The precise formla is

.. math::
    
    a_{i} = \sum_{j=1}^{j<i}{[S_{j}*a_{i}^{\prime}]}

Where

.. math::
    
    S_{i} = 1 - \sum_{j=1}^{j<i}{a_{j}}

Note that in this definition :math:`a_{1} = a_{1}^{\prime}` .

This rescaling makes it so that when setting bounds for these amplitudes in
minimize_ these amplitudes can be set to (0, 1), and the output distribution will maintain
either the integral for PDF, or final cumulative size for CDF like single distribution
functions.

``multi_..._cdfbins`` distributions
***********************************

The :func:`multi_cdfbins`, :func:`multi_cdfbins_free`, :func:`multi_nd_cdfbins`
functions all assume the input distribution is a CDF type-distribution.
The returned function assumes the ``x`` input is that of **bins** and thus will
return the expected *pmf* for the bins, of a size 1 less than the size of ``x``,
given as the difference in the CDF between bin edges.


``n<dist>_...`` functions
-------------------------

All functions named in the fashion ``n<dist>_...`` are creating from a ``multi_...``
function using a distribution from `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
These functions are named according to the following formula\:
``n<dist>_pdf/cdf/cdfbins`` the behavior of each is as follows:

    - ``n<dist>_pdf`` are build from :func:`multi_dist` and a pdf ``scipy.stats`` function.
      These have the signature ``n<dist>_pdf(param:np.ndarray, x:np.ndarray)``
      and return the PDF at the selected points of x.
      **Note** These functions shoudl be used with :func:`fit_column_mle`
    - ``n<dist>_cdf`` are build from :func:`multi_dist` and a cdf ``scipy.stats`` function.
      These have the signature ``n<dist>_cdf(param:np.ndarray, x:np.ndarray)``
      and return the CDF at the selected points of x.
      **Note** These functions should be used with :func:`fit_hist_cdf`
    - ``n<dist>_cdfbins`` are build from :func:`multi_cdfbins` and a cdf ``scipy.stats`` function.
      These have the signature ``n<dist>_cdfbins(param:np.ndarray, x:np.ndarray)``
      and return the *PMF* of the value *between* values of ``x`` 
      (i.e. difference between CDF of consecutive values of ``x``).
      **Note** This can be used to plot a fit vs the pmf of a histogram.
    - ``n<dist>_free` are build from :func:`multi_dist_free` and a PDF function.
      These have the signature ``n<dist>_free(param:np.ndarray, x:np.ndarray)``
      and return a non-normalized distribution. Notably these have a ``param``
      array that includes a final amp, and do *not* rescale the amplitudes.
      It should be noted that a PDF function is always used to create these distributions.
      Therefore the integral of these distributions will be the sum of the amplitudes.

``value_...`` and ``fit_...`` functions
---------------------------------------

The ``value_...`` functions are designed to be used as the ``func`` arg of minimize_
See each docstring for the expected ``args`` tuple to be supplied to minimize_
However, in general the last value in the tuple should be one of the ``n<dist>...``
functions.

.. note::
    
    Only certian pairings between ``value_...`` and ``n<dist>_...`` are valid.
    Make sure to use a propper pairing.
    In general, :func:`value_mle` and :func:`value_lsq` should always be paired 
    with ``n<dist>_pdf`` functions, 
    while :func:value_lsq_by_cdf` should be paried with ``n<dist>_cdf`` functions.


The ``fit_...`` functions are further wrappers around ``value_...`` functions that
direcly call minimize_ .


.. _leastsquare: `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`__
.. _minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _optimizeresult: `OptimizeResult <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`__
.. |Param| replace:: `Column <smfbursts.datamodel.tables.Param>`
.. |Column| replace:: `Column <smfbursts.datamodel.tables.Column>`
.. |GateGroup| replace:: `Column <smfbursts.datamodel.tables.GateGroup>`

"""
from collections.abc import Callable, Sequence, Iterator
from typing import Any, Literal
from weakref import WeakValueDictionary
from itertools import chain, repeat
from math import isqrt
from functools import wraps, partial

import numpy as np
from scipy import stats
from scipy.optimize import minimize, least_squares, OptimizeResult

from .utils import fjit, fnumba, iter_funcinput, _echo
from .tables import Column, GateGroup, DataS, DataSet

FitFunc = Callable[[np.ndarray[np.float64],np.ndarray[np.float64]],np.ndarray[np.float64]]


@fjit(fnumba.float64[:](fnumba.float64[:]))
def retrieve_amps(amps:np.ndarray[np.float64])->np.ndarray[np.float64]:
    r"""
    Convert a set of "rescaled" amplitudes,  all in range (0, 1), into a set of
    amplitudes that all sum to 1. This is done by treating each amplitude
    as the fraction of the "remaining" amplitude. The output has 1 more element
    than the input.
    
    This function is used in "fixed" multi-dist functions to convert amplitudes
    that are bounded from (0, 1) into a set of amplitudes that sum to 1.
    
    Rescaling can be thought of as treating the amplitude at a given index as the 
    fraction of the amount of "remaining available amplitude".

    Rescaling is defined as follows\:

    :math:`a_{i} = \sum_{j=1}^{j<i}{[S_{j}*a_{i}^{\prime}]}`

    Where

    :math:`S_{i} = 1 - \sum_{j=1}^{j<i}{a_{j}}`
    
    Where :math:`a_{i}` is the *true* amplitude (output), and :math:`a_{i}^{\prime}` 
    is the "rescaled" amplitude (input)
    
    .. note::
        
        This is the inverse function of :func:`rescale_amps`


    Parameters
    ----------
    amps : np.ndarray[np.float64]
        Amplitudes given as faction of remaining.

    Returns
    -------
    np.ndarray[np.float64]
        Actual amplitudes, will sum to 1.

    """
    rem = 1.0
    out = np.empty(amps.size + 1)
    for i in range(amps.size):
        out[i] = amps[i] * rem
        rem -= out[i]
    out[amps.size] = rem
    return out


@fjit(fnumba.float64[:](fnumba.float64[:]))
def rescale_amps(amps:np.ndarray[np.float64])->np.ndarray[np.float64]:
    """
    Rescale a set of raw/true amplitudes into a "rescaled" version.
    This function automatically normalizes the input amplitudes first,
    (i.e. the sum of amplitudes is 1), before rescaling.
    
    This function is used to take a set of raw amplitudes and convert them into
    the rescaled amplitudes that can be used in a "fixed" multi-dist function.
    
    Rescaling can be thought of as treating the amplitude at a given index as the 
    fraction of the amount of "remaining available amplitude".

    Rescaling in defined as follows\:
    
    :math:`A_{i} = \sum_{j=1}^{j<i}{[S_{j}*a_{i}]}`

    Where

    :math:`S_{i} = 1 - \sum_{j=1}^{j<i}{A_{j}}`
    
    Where :math:`A_{i}` is the *true* amplitude (input after dividing by sum), 
    and :math:`a_{i}` is the "rescaled" amplitude
        
    
    .. note::
        
        This is the inverse function of :func:`rescale_amps`


    Parameters
    ----------
    amps : np.ndarray[np.float64]
        Raw amplitudes of each subsituent distribution.

    Returns
    -------
    np.ndarray[np.float64]
        Rescaled amplitudes so that each amplitude is the fraction of the distribution
        to the remaining amplitude.

    """
    amp = amps / np.sum(amps)
    tot = 1.0
    out = np.empty(amps.size - 1)
    for i in range(out.size):
        out[i] = amp[i] / tot
        tot -= out[i]
    return out


def multi_dist(dist:Callable[[np.ndarray[np.float64],np.ndarray[np.float64]],np.ndarray],
               nfparam:int|Sequence[str], name:str=None, doc:str=None)->FitFunc:
    r"""
    Create an arbitrary "multi" distribution, but assuming the ``dist`` is a PDF-like function. 
    The key difference is that a) the last amplitude is omitted from the params array, 
    as it is calculated from the other amplitudes. 
    All amplitudes should be in the range (0,1) (double open).
    Successive amplitudes are the fraction of the *remaining* amplitide.
    This is to facilitate fitting algorithms with fixed bounds.
    Since the ampltidues in output must sum to 1, it is not enough that all "final"
    amplitudes are in the range of (0,1), but that their sum is 1.
    
    As an example, consider the 3 distribution case\:
    if the input array is ``p1, a1, p2, a2, p3 = [p1, 0.5, p2, 0.5, p3]``
    the actual amplitudes would be 
    ``A1 = 0.5, A2 = (1-0.5)*0.5=0.25, A3= 1-(0.5+0.25)=0.25``.
                                                  

    Parameters
    ----------
    dist : Callable[[np.ndarray[np.float64], np.ndarray[np.float64]],np.ndarray]
        PDF-like single distribution function.
    nfparam : int | Sequence[str]
        Number of parameter to dist.
    
    name : str, optional
        Name for funciton, only used in creating return value docstring.
        The default is None.

    Returns
    -------
    FitFunc
        PDF-like mutli-distribution functions.
        Takes signature ``fit(params:np.ndarray, x:np.ndarray, **kwargs)``.

    """
    if isinstance(nfparam, Sequence):
        param_names = tuple(nfparam)
        nfparam = len(param_names)
    else:
        param_names = tuple(f'p{i}' for i in range(nfparam))
    nstride = nfparam + 1
    name = dist.__name__ if name is None else name
    doc = '' if doc is None else doc
    def distfunc(params:np.ndarray, x:np.ndarray[np.float64], **kwargs)->np.ndarray[np.float64]:
        amps = retrieve_amps(params[nfparam::nfparam+1])
        return sum(a*dist(x, *params[nstride*l:nstride*l+nfparam], **kwargs) 
                   for l, a in enumerate(amps))
    distfunc._nfparam = nfparam
    distfunc._param_names = param_names
    distfunc._free = False
    distfunc.__doc__ = f"""
    Multi {name} distribution, with amplitudes rescaled by :func:`retrieve_amps` 
    So that function can be used in optimize where bounds for amplitudes
    can be set to ``(0.0, 1.0)``
    
    {doc}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining the the distribution. Organized as repeating array
        of ``[p0_1, ... p0_{nfparam}, amp0, ... pm_1, ... pm_{nfparam}]``
    x : np.ndarray[np.float64]
        Values where to compute distribution.
    **kwargs : Any
        Additional kwargs handed to {name}.

    Returns
    -------
    np.ndarray
        Value of distribution at each value of x.

    """
    return distfunc


def multi_dist_free(dist:Callable[[np.ndarray[np.float64], np.ndarray[np.float64]],np.ndarray], 
                    nfparam:int|Sequence[str], name:str=None, doc:str=None)->FitFunc:
    """
    Create an arbitrary "multi" distribution function.
    The return value is a callable that should take a ``params`` array and an
    `x` value, and returns the sum of ``a*dist(x, *param)`` where param is 
    a slice of the ``params`` array. The ``params`` array organized like
    ``p1_1, ... p1_m, a1, p2_1, ... p2_m, a2, ... pN_1, ... pN_m, aN``. m is 
    the value of `nfparam` and ``N`` is flexible. ``pN_m`` means the m-th parameter
    of the N-th distribution, and aN is the N-th amplitude of said parameter
    

    Parameters
    ----------
    dist : Callable[[np.ndarray[np.float64], np.ndarray[np.float64]],np.ndarray]
        "Single" distribution function from which to create a multi.
    nfparam : int  | Sequence[str]
        Number of parameters to dist.
    name : str, optional
        Name for funciton, only used in creating return value docstring.
        The default is None.
    doc : str, optional
        Additional docstring to add between introduction and Arguments of docstring.

    Returns
    -------
    FitFunc
        Multi-distribution function. Takes signature ``fit(params:np.ndarray, x:np.ndarray, **kwargs)``.

    """
    if isinstance(nfparam, Sequence):
        param_names = tuple(nfparam)
        nfparam = len(param_names)
    else:
        param_names = tuple(f'p{i}' for i in range(nfparam))
    nstride = nfparam + 1
    name = dist.__name__ if name is None else name
    doc = '' if doc is None else doc
    def distfunc(params:np.ndarray, x:np.ndarray[np.float64], **kwargs)->np.ndarray[np.float64]:
        amps = params[nfparam::nstride]
        return sum(a*dist(x, *params[nstride*l:nstride*l+nfparam], **kwargs) for l, a in enumerate(amps))
    distfunc._nfparam = nfparam
    distfunc._param_names = param_names
    distfunc._free = True
    distfunc.__doc__ = f"""
    Multi {name} distribution
    
    {doc}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining the the distribution. Organized as repeating array
        of ``[p0_1, ... p0_{nfparam}, amp0, ... pm_1, ... pm_{nfparam}, ampm]``
    x : np.ndarray[np.float64]
        Values where to compute distribution.
    **kwargs : Any
        Additional kwargs handed to {name}.

    Returns
    -------
    np.ndarray
        Value of distribution at each value of x.

    """
    return distfunc


def multi_cdfbins(dist:Callable[[np.ndarray,np.ndarray],np.ndarray[np.float64]], 
                  nfparam:int|Sequence[str], name:str=None, doc:str=None)->FitFunc:
    r"""
    Create an arbitrary "multi" distribution, assuming the ``dist`` is a PDF-like function. 
    The key difference is that a) the last amplitude is omitted from the params array, as it is calculated
    from the other amplitudes. All amplitudes should be in the range (0,1) (double open).
    Successive amplitudes are the fraction of the *remaining* amplitide.
    This is to facilitate fitting algorithms with fixed bounds.
    Since the ampltidues in output must sum to 1, it is not enough that all "final"
    amplitudes are in the range of (0,1), but that their sum is 1.
    
    As an example, consider the 3 distribution case\:
    if the input array is ``p1, a1, p2, a2, p3 = [p1, 0.5, p2, 0.5, p3]``
    the actual amplitudes would be 
    ``A1 = 0.5, A2 = (1-0.5)*0.5=0.25, A3= 1-(0.5+0.25)=0.25``.

    Parameters
    ----------
    dist : Callable[[np.ndarray[np.float64], np.ndarray[np.float64]],np.ndarray]
        PDF-like single distribution function.
    nfparam : int | Sequence[str]
        Number of parameter to dist.
    name : str, optional
        Name for funciton, only used in creating return value docstring.
        The default is None.\
    doc : str, optional
        Additional docstring to add between introduction and Arguments of docstring.

    Returns
    -------
    FitFunc
        PDF-like mutli-distribution functions.
        Takes signature ``fit(params:np.ndarray, x:np.ndarray, **kwargs)``.

    """
    if isinstance(nfparam, Sequence):
        param_names = tuple(nfparam)
        nfparam = len(param_names)
    else:
        param_names = tuple(f'p{i}' for i in range(nfparam))
    nstride = nfparam + 1
    name = dist.__name__ if name is None else name
    doc = '' if doc is None else doc
    def cdfbinfunc(params:np.ndarray, x:np.ndarray[np.float64], **kwargs)->np.ndarray[np.float64]:
        amps = retrieve_amps(params[nfparam::nfparam+1])
        return np.diff(sum(a*dist(x, *params[nstride*l:nstride*l+nfparam], **kwargs)
                           for l, a in enumerate(amps)))
    cdfbinfunc._nfparam = nfparam
    cdfbinfunc._param_names = param_names
    cdfbinfunc._free = False
    cdfbinfunc.__doc__ = f"""
    Multi {name} distribution, with amplitudes rescaled by :func:`retrieve_amps` 
    and x is assumed to be bin edges, evaluating the difference in cdf between
    bins.
    So that function can be used in optimize where bounds for amplitudes
    can be set to ``(0.0, 1.0)``
    
    {doc}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining the the distribution. Organized as repeating array
        of ``[p0_1, ... p0_{nfparam}, amp0, ... pm_1, ... pm_{nfparam}]``
    x : np.ndarray[np.float64]
        Values of bin edges at which to compute the distribution.
    **kwargs : Any
        Additional kwargs handed to {name}.

    Returns
    -------
    np.ndarray
        Value of distribution at each value of x.

    """
    return cdfbinfunc


def multi_cdfbins_free(dist:Callable[[np.ndarray[np.float64], np.ndarray[np.float64]],np.ndarray], 
                       nfparam:int|Sequence[str], name:str=None, doc:str=None)->FitFunc:
    """
    Create an arbitrary "multi" distribution function, params are not rescaled.
    The return value is a callable that should take a ``params`` array and an
    `x` value, and returns the sum of ``a*dist(x, *param)`` where param is 
    a slice of the ``params`` array. The ``params`` array organized like
    ``p1_1, ... p1_m, a1, p2_1, ... p2_m, a2, ... pN_1, ... pN_m, aN``. m is 
    the value of `nfparam` and ``N`` is flexible. ``pN_m`` means the m-th parameter
    of the N-th distribution, and aN is the N-th amplitude of said parameter.

    Parameters
    ----------
    dist : Callable[[np.ndarray[np.float64], np.ndarray[np.float64]],np.ndarray]
        "Single" distribution function from which to create a multi.
    nfparam : int | Sequence[str]
        Number of parameters to dist.
    name : str, optional
        Name for funciton, only used in creating return value docstring.
        The default is None.
    doc : str, optional
        Additional docstring to add between introduction and Arguments of docstring.

    Returns
    -------
    FitFunc
        Multi-distribution function. Takes signature ``fit(params:np.ndarray, x:np.ndarray, **kwargs)``.

    """
    if isinstance(nfparam, Sequence):
        param_names = tuple(nfparam)
        nfparam = len(param_names)
    else:
        param_names = tuple(f'p{i}' for i in range(nfparam))
    nstride = nfparam + 1
    name = dist.__name__ if name is None else name
    doc = '' if doc is None else doc
    def distfunc(params:np.ndarray, x:np.ndarray[np.float64], **kwargs)->np.ndarray[np.float64]:
        amps = params[nfparam::nstride]
        return np.diff(sum(a*dist(x, *params[nstride*l:nstride*l+nfparam], **kwargs)
                   for l, a in enumerate(amps)))
    distfunc._nfparam = nfparam
    distfunc._param_names = param_names
    distfunc._free = True
    distfunc.__doc__ = f"""
    Multi {name} distribution, computed based on bin edges of cdf.
    
    {doc}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining the the distribution. Organized as repeating array
        of ``[p0_1, ... p0_{nfparam}, amp0, ... pm_1, ... pm_{nfparam}, ampm]``
    x : np.ndarray[np.float64]
        Bin edges of where to compute distribution 
        (output array has 1 fewer elements compared to x).
    **kwargs : Any
        Additional kwargs handed to {name}.

    Returns
    -------
    np.ndarray
        Value of distribution at each value of x.

    """
    return distfunc




def mnd_func(nfunc:Callable[[int],int], infunc:Callable[[int],int]=None, inv:Callable[[np.ndarray],np.ndarray]=None):
    """
    Decorator to create a ``mnd_...`` type function, for use specifying nfparam
    fucntion in :func:`multi_dist_nd` and :func:`multi_cdfbins_nd` functions.

    Parameters
    ----------
    nfunc : Callable[[int],int]
        Function which takes number of dimensions and specifies number of parameter
        values for the given parameter (size of parameter array).
    infunc : Callable[[int],int], optional
        Inverse function of nfunc, should take size of parameter array and return
        number of dimensions. The default is None.
    inv : Callable[[np.ndarray],np.ndarray], optional
        Inverse of function to be decorated, should take array for input to
        distribution and return the flattened array for concatenation into parameters. 
        The default is None.

    
    """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._nfunc = nfunc
        wrapper._infunc = infunc
        wrapper._invparam = inv
        return wrapper
    return inner


def _ret1(nd:int)->int:
    """always returns 1, used for mnd_scalar"""
    return 1


def _retn(nd:int)->int:
    """returns nd, used for mnd_scalar, mnd_covar_zero"""
    return nd


@mnd_func(_ret1, lambda n: None, lambda arr: np.atleast_1d(arr))
def mnd_scalar(param:np.ndarray)->float:
    """Mnd function for scalar values (does not depend on number of arrays in nd distribution)"""
    return param[0]


@mnd_func(_retn, _retn, _echo)
def mnd_vec(param:np.ndarray, nd:int)->np.ndarray:
    """Mnd function for vector values (same number of parameters as number of arrays)"""
    return param

@mnd_func(lambda n: n*(n+1) // 2, lambda n: isqrt(8*n+1) // 2, lambda arr: arr[np.tri(arr.shape[0],dtype=np.bool_)].reshape(-1))
def mnd_covar(param:np.ndarray, nd:int)->np.ndarray:
    """Mnd function for covariance (symetric positive definite matrix), provide upper diagonal"""
    cov = np.empty((nd, nd))
    p = 0
    for i in range(nd):
        cov[i,i:] = param[p:p+nd-i]
        p += nd - i
    m = np.tri(nd, dtype=np.bool_)
    cov[m] = cov.T[m]
    return cov


@mnd_func(_retn, _retn, lambda arr: np.diagonal(arr))
def mnd_covar_zero(param:np.ndarray, nd:int)->np.ndarray:
    """Mnd function for diagonal matrix value param, (supply diagonal, same number of params as number of arrays)"""
    return np.diagflat(param)



@mnd_func(lambda n: n**2, lambda n: isqrt(n), lambda arr: arr.reshape(-1))
def mnd_mat(param:np.ndarray, nd:int)->np.ndarray:
    """Mnd function for square matrix (N^2 size param for number of arrays)"""
    return param.reshape(nd, nd)


NdNfunc = Literal['scalar','vec', 'covar_zero', 'covar','mat']|Callable[[np.ndarray, int], np.ndarray]
_mnd_fmap = {'scalar':mnd_scalar, 'vec':mnd_vec, 'covar_zero':mnd_covar_zero, 'covar':mnd_covar, 'mat':mnd_mat}


def _nd_diff(hst:np.ndarray)->np.ndarray:
    """
    Compute sequential difference along each axis of hst. Useful for computing
    nd pdf bins from cdf.

    Parameters
    ----------
    hst : np.ndarray
        Nd array to compute difference.

    Returns
    -------
    np.ndarray[np.float64]
        Successive difference between elements along each axis.

    """
    for i in range(hst.ndim):
        hst = np.diff(hst, axis=i)
    return hst


def _nd_csum(hst:np.ndarray)->np.ndarray:
    """
    Compute cumulative sum along each axis of hst. Useful for computing
    nd cdf bins from pmf.

    Parameters
    ----------
    hst : np.ndarray
        Nd array to compute difference.

    Returns
    -------
    np.ndarray[np.float64]
        Successive sum between elements along each axis.

    """
    for i in range(hst.ndim):
        hst = np.cumsum(hst, axis=i)
    return hst


NdFitFunc = Callable[[Sequence[np.ndarray[np.float64]],np.ndarray[np.float64]],np.ndarray[np.float64]]
NdDistFunc = Callable[[np.ndarray[np.float64],Sequence[np.ndarray[np.float64]]],np.ndarray]


def _nparam_size(distfunc:NdFitFunc, n:int):
    """Compute numper of parameter values per distribution of nd distribuion based on number of dimensions"""
    return np.cumsum([0]+[mnd._nfunc(n) for mnd in distfunc._ndfuncs])


def _check_nfparams(params:np.ndarray[np.float64], nparams:np.ndarray, naparam:int)->np.ndarray[np.float64]:
    """Check if nd parameter (non-free) array is valid size and return true amps"""
    if params.size % naparam != nparams[-1]:
        raise ValueError(f"cannot unpack param of size {params.size} into multi-dist, single dist size {nparams[-1]}")
    return retrieve_amps(params[nparams[-1]::naparam])


def _check_nfparams_free(params:np.ndarray[np.float64], nparams:np.ndarray, naparam:int)->np.ndarray[np.float64]:
    """Check if nd free parameter array is valid size and return true amps"""
    if params.size % naparam != 0:
        raise ValueError(f"cannot unpack param of size {params.size} into multi-dist, single dist size {nparams[-1]}")
    return params[nparams[-1]::naparam]


def multi_dist_nd(dist:NdDistFunc, params:Sequence[NdNfunc], name:str=None, 
                  free:bool=False, param_names:Sequence[str]=None, doc:str=None)->NdFitFunc:
    """
    Create a multi-dimensional multi-distribution function. Output function
    will take signature ``nd<dist>_...(params:np.ndarray, x:np.ndarray)``
    Where params is organized similar to that of :func:`multi_dist`, however,
    each sub-param is determined by the size of the last dimension of x.
    

    Parameters
    ----------
    dist : NdDistFunc
        Function of single distribution.
    params : NdNfunc
        Sequence of parameter array definitions, either ``mnd_...`` function or
        one of the following: 'scalar','vec', 'covar_zero', 'covar','mat'.
    name : str, optional
        Name of distribution. The default is None.
    free : bool, optional
        Whether function has n-amplitudes and no amplitude regularization (True)
        or if function has n-1 amplitudes and rescales them (False). The default is False.
    param_names : Sequence[str], optional
        Names given to each parameter array. The default is None.

    Returns
    -------
    NdFitFunc
        Variable dimensional multi-distribution.

    """
    params = tuple(_mnd_fmap.get(param, param) for param in params)
    param_names = tuple(f'p{i}' for i in range(len(params))) if param_names is None else tuple(param_names)
    name = dist.__name__ if name is None else name
    free = bool(free)
    param_check = _check_nfparams_free if free else _check_nfparams
    free_str = '"free"' if free else ''
    rescale_str = "1 amplitude per distribution" if free else "Rescales amplitudes so that bounds can be set from (0, 1)."
    final_amp = ', ampN' if free else ''
    ret_str = "Effective pmf, size along each dimension 1 less than each bin" if free else "Value of distribution at each value of x."
    doc = '' if doc is None else doc
    def distfunc(params:np.ndarray[np.float64], x:np.ndarray[np.float64])->np.ndarray[np.float64]:
        nd = x.shape[-1]
        nparams = _nparam_size(distfunc, nd)
        naparam = nparams[-1] + 1
        amps = param_check(params, nparams, naparam)
        return sum(a*dist(x, *(nf(params[i*naparam+pb:i*naparam+pe], nd) 
                                 for nf, pb, pe in zip(distfunc._ndfuncs, nparams[:-1], nparams[1:])))
                   for i, a in enumerate(amps))
    distfunc._ndfuncs = params
    distfunc._free = free
    distfunc._param_names = param_names
    distfunc.__doc__ = f"""
    Nd multi {name} {free_str} distribution. {rescale_str}
    
    {doc}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining the the distribution. Organized as repeating array
        of ``<params0>, amp0, ...<paramsN>{final_amp}``
    x : np.ndarray[np.float64]
        Array of locations to evaluate. Last dimension is nd vector, other dimensions
        maintained.
    **kwargs : Any
        Additional kwargs handed to {name}.

    Returns
    -------
    np.ndarray
        {ret_str}

    """
    return distfunc


def multi_cdfbins_nd(dist:NdDistFunc, params:NdNfunc, name:str=None, 
                  free:bool=False, param_names:Sequence[str]=None, doc:str=None)->NdFitFunc:
    params = tuple(_mnd_fmap.get(param, param) for param in params)
    param_names = tuple(f'p{i}' for i in range(len(params))) if param_names is None else tuple(param_names)
    name = dist.__name__ if name is None else name
    free = bool(free)
    param_check = _check_nfparams_free if free else _check_nfparams
    free_str = '"free"' if free else ''
    rescale_str = "1 amplitude per distribution" if free else "Rescales amplitudes so that bounds can be set from (0, 1)."
    final_amp = ', ampN' if free else ''
    ret_str = "Effective pmf, size along each dimension 1 less than each bin" if free else "Value of distribution at each value of x."
    doc = '' if doc is None else doc
    def distfunc(params:np.ndarray[np.float64], bins:np.ndarray[np.float64])->np.ndarray[np.float64]:
        nd = bins.shape[-1]
        nparams = _nparam_size(distfunc, nd)
        naparam = nparams[-1] + 1
        amps = param_check(params, nparams, naparam)
        return _nd_diff(sum(a*dist(bins, *(nf(params[i*naparam+pb:i*naparam+pe], nd)
                                           for nf, pb, pe in zip(distfunc._ndfuncs, nparams[:-1], nparams[1:])))
                            for i, a in enumerate(amps)))
    distfunc._ndfuncs = params
    distfunc._free = free
    distfunc._param_names = param_names
    distfunc.__doc__ = f"""
    Nd multi {name} {free_str} distribution showing the pmf for bins. {rescale_str}
    
    {doc}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining the the distribution. Organized as repeating array
        of ``<params0>, amp0, ...<paramsN>{final_amp}``
    bins : Sequence[np.ndarray[np.float64]]
        Sequence of arrays given as first argument to {name}, should be same length,
        Nth array is Nth dimension of vector X.
    **kwargs : Any
        Additional kwargs handed to {name}.

    Returns
    -------
    np.ndarray
        {ret_str}

    """
    return distfunc


def value_mle(params:np.ndarray, samples:np.ndarray, func:FitFunc, **kwargs)->float:
    r"""
    Compute the negative log Maximum Likelihood Estimator value for a multi-distribution ``func``
    with parameters ``params`` for a group os samples ``samples``.
    
    .. math::
        
        P(X | \theta) = MLE = \prod_{i=1}^{N}{PDF_{\theta}(\vec{x_{i}})}
    
    For numerical accuracy and use in minimize_ function, computes as
    
    .. math::
        
        -log(MLE) = -\sum_{i=1}^{N}{log(PDF(\vec{x_{i}}))}

    Parameters
    ----------
    params : np.ndarray
        Parameters defining multi distribution.
    samples : np.ndarray
        Sample values to compute MLE.
    func : FitFunc
        PDF-like multi-distribution function.
    **kwargs : Any
        Additional kwargs passed to func.

    Returns
    -------
    float
        negative log MLE of distribtuion given data.

    """
    return -np.log(func(params, samples, **kwargs)).sum()


def _value_diff(params:np.ndarray, x:np.ndarray, y:np.ndarray, func:FitFunc, **kwargs)->np.ndarray:
    """Internal function for :func:`value_diff`, does not reshape output to be 1D"""
    return y - func(params, x, **kwargs)


def value_diff(params:np.ndarray, x:np.ndarray, y:np.ndarray, func:FitFunc, **kwargs)->np.ndarray:
    """
    Compute difference between measured distribution defined by
    :math:`f(x) = y`, and a multi-distribution defined by ``params`` and ``func``.
    Typically ``func`` is a multi-PDF like distribution, therefore ``y`` should
    be normalize such that it is PDF-like (if histogram this means pmf times binwidths).
    
    The final evaluation is ``((y - func(params, x, **kwargs))**2).sum()``

    Parameters
    ----------
    params : np.ndarray
        param values for func.
    x : np.ndarray
        x values of disribution, will be .
    y : np.ndarray
        y values of distribution.
    func : FitFunc
        multi-distribution (should be PDF-like).
    **kwargs : Any
        Additional kwargs handed to func.

    Returns
    -------
    np.ndarray
        Array of differences between dist at x given params and y.

    """
    return _value_diff(params, x, y, func, **kwargs).reshape(-1)


def value_lsq(params:np.ndarray, x:np.ndarray, y:np.ndarray, func:FitFunc, **kwargs)->float:
    """
    Compute least-square difference between measured distribution defined by
    :math:`f(x) = y`, and a multi-distribution defined by ``params`` and ``func``.
    Typically ``func`` is a multi-PDF like distribution, therefore ``y`` should
    be normalize such that it is PDF-like (if histogram this means pmf times binwidths).
    
    The final evaluation is ``((y - func(params, x, **kwargs))**2).sum()``

    Parameters
    ----------
    params : np.ndarray
        param values for func.
    x : np.ndarray
        x values of disribution, will be .
    y : np.ndarray
        y values of distribution.
    func : FitFunc
        multi-distribution (should be PDF-like).
    **kwargs : Any
        Additional kwargs handed to func.

    Returns
    -------
    float
        Least-square difference between dist at x given params and y.

    """
    return (_value_diff(params, x, y, func, **kwargs)**2).sum()


def _value_diff_by_cdf(params:np.ndarray, bins:np.ndarray|Sequence[np.ndarray], y:np.ndarray, func:FitFunc, **kwargs)->np.ndarray:
    """Internal function for :func:`value_diff_by_cdf`, does not reshape array"""
    expect = func(params, bins, **kwargs)
    expect -= expect[tuple(0 for _ in range(expect.ndim))]
    expect = expect / expect[tuple(-1 for _ in range(expect.ndim))]
    return y - _nd_diff(expect)


def value_diff_by_cdf(params:np.ndarray, bins:np.ndarray|Sequence[np.ndarray], y:np.ndarray, func:FitFunc, **kwargs)->np.ndarray:
    r"""
    Compute difference between measured distribution (``y``) and the pmf 
    calculated from the ``bin`` edges of a CDF function.
    Expects ``func`` to be a ``n<dist>_cdf`` or ``nd<dist>_cdf`` like function.
    Note that ``bins`` length of each bins array should be 1 larger than equivalent
    dimension in ``y``. Also note that output of func is rescaled so that values
    range from 0 to 1, ie the range of ``x`` -bins is assumed to be the "full" CDF.
    This is because it is assumed that ``y`` is the "full" pmf.
    

    Parameters
    ----------
    params : np.ndarray
        Params of nd multi-distribution.
    bins : np.ndarray | Sequence[np.ndarray]
        Bins of distribution.
    y : np.ndarray
        Observed pmf of histogram along bins.
    func : FitFunc
        ``n<dist>_cdf`` or ``nd<dist>_cdf`` like function defining the distribution.
    **kwargs : Any
        Additional parmeters handed to ``func``.

    Returns
    -------
    np.ndarray
        Array of differences between computed pmf and y.

    """
    return _value_diff_by_cdf(params, bins, y, func).reshape(-1)


def value_lsq_by_cdf(params:np.ndarray, bins:np.ndarray|Sequence[np.ndarray], y:np.ndarray, 
                     func:FitFunc, **kwargs)->float:
    r"""
    Compute least-square difference between measured distribution (``y``) and the
    pmf calculated from the ``bin`` edges of a CDF function.
    Expects ``func`` to be a ``n<dist>_cdf`` or ``nd<dist>_cdf`` like function.
    Note that ``bins`` length of each bins array should be 1 larger than equivalent
    dimension in ``y``. Also note that output of func is rescaled so that values
    range from 0 to 1, ie the range of ``x`` -bins is assumed to be the "full" CDF.
    This is because it is assumed that ``y`` is the "full" pmf.
    

    Parameters
    ----------
    params : np.ndarray
        Params of nd multi-distribution.
    bins : np.ndarray | Sequence[np.ndarray]
        Bins of distribution.
    y : np.ndarray
        Observed pmf of histogram along bins.
    func : FitFunc
        ``n<dist>_cdf`` or ``nd<dist>_cdf`` like function defining the distribution.
    **kwargs : Any
        Additional parmeters handed to ``func``.

    Returns
    -------
    float
        Least-square difference between computed pmf and y.

    """
    return (_value_diff_by_cdf(params, bins, y, func)**2).sum()


# 1d distributions 
ngaus_pdf = multi_dist(stats.norm.pdf, ['mu', 'sigma'], name='normal.pdf')
nbeta_pdf = multi_dist(stats.beta.pdf, ['a', 'b'], name='beta.pdf')
nexponnorm_pdf = multi_dist(stats.exponnorm.pdf, ['K',], name='exponnorm.pdf')

ngaus_cdf = multi_dist(stats.norm.cdf, ['mu', 'sigma'], name='normal.cdf')
nbeta_cdf = multi_dist(stats.beta.cdf, ['a', 'b'], name='beta.cdf')
nexponnorm_cdf = multi_dist(stats.exponnorm.cdf, ['K',], name='exponnorm.cdf')

ngaus_free = multi_dist_free(stats.norm.pdf, ['mu', 'sigma'], name='normal.pdf')
nbeta_free = multi_dist_free(stats.beta.pdf, ['a', 'b'], name='beta.pdf')
nexponnorm_free = multi_dist_free(stats.exponnorm.pdf, ['K',], name='exponnorm.pdf')

ngaus_cdfbins = multi_cdfbins(stats.norm.cdf, ['mu', 'sigma'], name='normal.pdf')
nbeta_cdfbins = multi_cdfbins(stats.beta.cdf, ['a', 'b'], name='beta.pdf')
nexponnorm_cdfbins = multi_cdfbins(stats.exponnorm.cdf, ['K',], name='exponnorm.pdf')


# Nd distributions
ndgaus_pdf = multi_dist_nd(stats.multivariate_normal.pdf, (mnd_vec, mnd_covar), 
                           name='multivariate_normal.pdf', param_names=['mu', 'sigma'])
ndgaus_zerocovar_pdf = multi_dist_nd(stats.multivariate_normal.pdf, (mnd_vec, mnd_covar_zero), 
                                     name='multivariate_normal.pdf', param_names=['mu', 'sigma'])


ndgaus_cdf = multi_dist_nd(stats.multivariate_normal.cdf, (mnd_vec, mnd_covar), 
                           name='multivariate_normal.cdf', param_names=['mu', 'sigma'])
ndgaus_zerocovar_cdf = multi_dist_nd(stats.multivariate_normal.cdf, (mnd_vec, mnd_covar_zero), 
                                     name='multivariate_normal.cdf', param_names=['mu', 'sigma'])

ndgaus_cdfbins = multi_cdfbins_nd(stats.multivariate_normal.cdf, (mnd_vec, mnd_covar), 
                                  name='multivariate_normal.cdf', param_names=['mu', 'sigma'])
ndgaus_zerocovar_cdfbins = multi_cdfbins_nd(stats.multivariate_normal.cdf, (mnd_vec, mnd_covar_zero), 
                                            name='multivariate_normal.cdf', param_names=['mu', 'sigma'])

ndgaus_free = multi_dist_nd(stats.multivariate_normal.pdf, (mnd_vec, mnd_covar), free=True,
                            name='multivariate_normal.pdf', param_names=['mu', 'sigma'])
ndgaus_zerocovar_free = multi_dist_nd(stats.multivariate_normal.pdf, (mnd_vec, mnd_covar_zero), free=True,
                                      name='multivariate_normal.pdf', param_names=['mu', 'sigma'])


def multinomial_ratio(x:np.ndarray[np.int64], params:np.ndarray[np.float64], )->np.ndarray[np.float64]:
    """Single multinomial distribution with variable total number of counts"""
    return stats.multinomial.pmf(x, x.sum(axis=-1), params)


@mnd_func(lambda  n: n - 1, lambda n : n + 1, rescale_amps)
def mnd_vec_norm(param:np.ndarray[np.float64], nd:int)->np.ndarray[np.float64]:
    """Mnd function for stochastic vector"""
    return retrieve_amps(param)

ndmultinomial_ratio_pmf = multi_dist_nd(multinomial_ratio, (mnd_vec_norm, ), name='multinomial ratio', param_names=('p', ))


def _mesharrays(*args:np.ndarray)->np.ndarray:
    out = np.empty([a.size for a in args]+[len(args),], dtype=args[0].dtype)
    for i, a in enumerate(args):
        out[...,i] = a.reshape(*(1 if j != i else a.size for j in range(len(args))))
    return out


class Hist:
    """
    Multi-dimensional histogram, with properties for evaluating the pdf, pmf,
    and binwidths.
    
    Parameters
    ----------
    hist : np.ndarray[np.int64]
        N-d array histogram
    bins : tuple[np.ndarray[np.float64]]
        N length tuple of 1D arrays defining bins of histogram. Size of N-th array
        should be 1 greater than N-th dimension of hist
    source : tuple[DataS, tuple[Column,...]], optional
        Definition of "source" of histogram, ie the data object and columns being
        histogram. This is not actively checked, and may be blank. 
        The default is None.
    
    """
    def __init__(self, hist:np.ndarray[np.int64], bins:Sequence[np.ndarray[np.float64]], source:tuple[DataS, Sequence[Column]]=None):
        if isinstance(bins, np.ndarray) and bins.ndim == 1:
            bins = (bins, )
        else:
            bins = tuple(np.asarray(b) for b in bins)
        hist = np.asarray(hist)
        if len(bins) != hist.ndim:
            raise ValueError(f"bins and histogram have inconsistent dimensions: {len(bins)} vs {hist.ndim}")
        for i, (b, h) in enumerate(zip(bins, hist.shape)):
            if b.ndim != 1:
                raise ValueError("bins arrays must be 1-D")
            if b.size -1 != h:
                raise ValueError(f"Size of histogram along dimension {i} ({h}) expected ({b.size-1})")
        self._bins = bins
        self._hist = hist
        self._sum = hist.sum()
        self._source = source
        self._cache = WeakValueDictionary()
    
    @property
    def bins(self)->np.ndarray[np.float64]:
        """Bins of histogram"""
        return self._bins
    
    @property
    def ndim(self)->int:
        """Number of dimensions in histogram (nd of hist array)"""
        return len(self._bins)
    
    @property
    def shape(self)->tuple[int,...]:
        """Shape of hist"""
        return self._hist.shape
    
    @property
    def bincenters(self)->list[np.ndarray]:
        """Sequence of bin centers for each dimension"""
        if not hasattr(self, '_bincenters'):
            self._bincenters = tuple(b[:-1] + np.diff(b)/2 for b in self._bins)
        return self._bincenters

    @property
    def hist(self)->np.ndarray[np.int64]:
        """Base histogram, counts per bin"""
        return self._hist

    @property
    def pmf(self)->np.ndarray[np.float64]:
        """Values of histogram where sum of all bins is 1"""
        if 'pmf' in self._cache:
            return self._cache['pmf']
        pmf = self._hist / self._sum
        self._cache['pmf'] = pmf
        return pmf
    
    @property
    def cdf(self)->np.ndarray[np.float64]:
        """Cumulative distribution function, probability that value is < (i, ...) of bins"""
        if 'cdf' in self._cache:
            return self._cache['cdf']
        cdf = _nd_csum(self.pmf)
        self._cache['cdf'] = cdf
        return cdf
    
    @property
    def binwidths(self)->tuple[np.ndarray[np.float64],...]:
        """Tuple of widths of bins along each dimension"""
        return tuple(np.diff(bins) for bins in self._bins)
    
    @property
    def binlocs(self)->np.ndarray[np.float64]:
        """
        Array of the location of each binedge in last dimenstion. 
        i.e. [...,i] is the position of of [...] in the ith dimension.
        Has nd + 1 dimensions, last dimension is size nd, shape of other dimensions
        is 1 greater than shape of histogram.
        """
        
        if 'binlocs' in self._cache:
            return self._cache['binlocs']
        out = _mesharrays(*self._bins)
        self._cache['binlocs'] = out
        return out
    
    @property
    def bincenterlocs(self)->np.ndarray[np.float64]:
        """
        Array of the location of each bin center in last dimenstion. 
        i.e. [...,i] is the position of of [...] in the ith dimension.
        Has nd + 1 dimensions, last dimension is size nd, other dimensions match
        shape of histogram.

        """
        if 'bincenterlocs' in self._cache:
            return self._cache['bincenterlocs']
        out = _mesharrays(*self.bincenters)
        self._cache['bincenterlocs'] = out
        return out
    
    @property
    def binsizes(self)->np.ndarray:
        """Array of same shape as hist, giving the product of binwidths"""
        if 'binsizes' in self._cache:
            return self._cache['binsizes']
        bws = np.ones(tuple(b.size-1 for b in self._bins))
        for i, b in enumerate(self._bins):
            bw = np.diff(b)
            bws *= bw.reshape(tuple(1 if i != j else -1 for j in range(len(self._bins))))
        self._cache['binsizes'] = bws
        return bws

    @property
    def pdf(self)->np.ndarray[np.float64]:
        """
        Probability density of each bin, 
        ie :attr:`Hist.pmf` divided by :attr:`Hist.binsizes`
        """
        if 'pdf' in self._cache:
            return self._cache['pdf']
        pdf = self.pmf / self.binsizes
        self._cache['pdf'] = pdf
        return pdf
    
    @property
    def logpdf(self)->np.ndarray[np.float64]:
        """Natural log of probability density of each bin"""
        if 'logpdf' in self._cache:
            return self._cache['logpdf']
        logpdf = np.log(self.pdf)
        self._cache['logpdf'] = logpdf
        return logpdf
    
    @classmethod
    def from_values(cls, sample:Sequence[np.ndarray], source:Any=None, **kwargs)->"Hist":
        """
        Create an nd histogram from n arrays. Wrapper around 
        `np.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`_

        Parameters
        ----------
        sample : Sequence[np.ndarray]
            Sequence of arrays of values to histogram.
        source : Any, optional
            Not used in computation, and indicator of the source of the data. 
            The default is None.
        **kwargs : Any
            Additional kwargs handed to 
            `np.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`_ .

        Returns
        -------
        "Hist"
            Histogram of sample.

        """
        return cls(*np.histogramdd(sample, **kwargs), source=source)
    
    @classmethod
    def from_columns(cls, origin:DataS, columns:Sequence[Column], gate:GateGroup=None, **kwargs)->"Hist":
        """
        Create an nd histogram from n-:class:`Column` s from origin data.

        Parameters
        ----------
        origin : DataS
            Data from which to get the columns.
        columns : Sequence[Column]
            Columns to histogram.
        gate : GateGroup, optional
            Gate to apply to all columns. The default is None.
        **kwargs : Any
            Kwargs handed to np.histogramdd.

        Returns
        -------
        Hist
            Histogram of columns.

        """
        columns = (columns, ) if isinstance(columns, Column) else columns
        func = origin.get_column if isinstance(origin, DataSet) else origin.concatenate_column
        if gate is None:
            for col in columns:
                gate = col.base_gate if gate is None else gate & col.base_gate
        carrs = tuple(func(col, gate) for col in columns)
        if 'bins' in kwargs and isinstance(kwargs['bins'], np.ndarray):
            kwargs['bins'] = np.atleast_2d(kwargs['bins'])
        return cls(*np.histogramdd(carrs, **kwargs), source=(origin, columns))


def _init_paramargs(func:FitFunc|None, args:tuple[np.ndarray[np.float64]], 
                    kwargs:dict[str,np.ndarray[np.float64]])->tuple[np.ndarray,...]|Iterator[np.ndarray]:
    """Process args/kwargs for ``n(d)<dist>_...``"""
    if kwargs:
        if not hasattr(func, '_param_names'):
            raise ValueError('{None if func is None else func.__name__} does not speficy names to match to kwargs')
        args = (arg for _, arg in iter_funcinput(func._param_names, {}, func._param_names, *args, **kwargs))
    return args



def _init_makeparam(func:FitFunc|None, args:tuple[np.ndarray[np.float64]], 
                    kwargs:dict[str,np.ndarray[np.float64]], free:None|bool
                    )->tuple[Sequence[np.ndarray[np.float64]],int,int,int]:
    """Get args arrays and check value sizes for make_init/bounds functions"""
    args =tuple(np.asarray(arg) for arg in _init_paramargs(func, args, kwargs))
    if func is not None and len(args) != func._nfparam:
        raise ValueError("Mismatched number of arguments to expected from {func.__name__}")
    if any(args[0].shape[0] != arg.shape[0] for arg in args[1:]):
        raise ValueError("all parameter/bounds arrays must have same number of distributions")
    free = func._free if hasattr(func, "_free") else free
    return args, len(args), args[0].shape[0], -int(not bool(free))
        


def make_init(*args:np.ndarray[np.float64], amps:np.ndarray[np.float64]=None, 
              free:bool=False, func:FitFunc=None, **kwargs:np.ndarray[np.float64])->np.ndarray[np.float64]:
    """
    Create an "init" param for use with mulit-dist functions, based on arrays
    of single param-types. Function wraps them together into single param array.
    
    Example usage:
    
    >>> mu = np.array([0.3, 0.7])
    >>> sigma = np.array([0.15, 0.15])
    >>> param = multifit.make_init(mu, sigma, func=multifit.ngaus_pdf)

    Parameters
    ----------
    *args : np.ndarray[np.float64]
        Arrays of different param types to wrap together into single params array.
    amps : np.ndarray[np.float64], optional
        Applitudes for each distribution, if not specified, amplitudes will all
        be equal after rescaling. The default is None.
    free : bool, optional
        If amplitudes are free or not. The default is False.
    func : FitFunc, optional
        ``n<dist>...`` function to check params agains. The default is None.
    **kwargs : np.ndarray[np.float64]
        kwargs named by parameters of `func`, arrays of given paramter value.

    Raises
    ------
    ValueError
        Mismatched shapes of input arrays.

    Returns
    -------
    init : np.ndarray
        Array for use in ``n<dist>_pdf`` function and optimization with ``fit_...`` function.

    """
    args, nparam, ndist, adjust = _init_makeparam(func, args, kwargs, free)
    amps = np.ones(ndist) if amps is None else np.asarray(amps, dtype=np.float64)
    if amps.size != ndist:
        raise ValueError("wrong size of amps compared to other parameters")
    if adjust:
        amps = rescale_amps(amps)
    naparam = nparam + 1
    init = np.empty((naparam*ndist+adjust))
    for i, arg in enumerate(chain(args, (amps, ))):
        init[i::naparam] = arg
    return init


def make_bounds(*args:np.ndarray[np.float64], free:bool=False, func:FitFunc=None, **kwargs:np.ndarray[np.float64])->np.ndarray[np.float64]:
    """
    Create a bounds Nx2 shapped array for use with minimize_ and ``fit_...`` functions.

    Parameters
    ----------
    *args : np.ndarray[np.float64]
        Arrays of bounds per param types to wrap into bounds, should have shape Nx2.
    free : bool, optional
        If distribution has "free" amplitudes. The default is False.
    func : FitFunc, optional
        `n<dist>_...` function to check bounds against. The default is None.
    **kwargs : np.ndarray[np.float64]
        kwargs named by parameters of `func`, arrays of bounds for a given paramter value.

    Raises
    ------
    ValueError
        Mismatched size of bounds arrays.

    Returns
    -------
    init : np.ndarray
        Nx2 bounds array for use with minimize_ .

    """
    args, nparam, ndist, adjust = _init_makeparam(func, args, kwargs, free)
    amps = np.zeros((ndist+adjust, 2))
    amps[:,1] = 1.0
    naparam = nparam + 1
    bounds = np.empty((naparam*ndist+adjust, 2))
    for i, arg in enumerate(chain(args, (amps,))):
        bounds[i::naparam,:] = arg
    return bounds


def unwrap_param(func:FitFunc, param:np.ndarray[np.float64], as_dict:bool=True
                 )->Sequence[np.ndarray[np.float64]]|dict[str:np.ndarray[np.float64]]:
    """
    Separate parameter array of ``n<dist>_...`` type array. 
    Behaves similar to inverse of :func:`make_init`
    
    >>> mr = fit_column_mle(data, column, multifit.ngaus_pdf, init)
    >>> mu, sigma, amps = unwrap_param(multifit.ngaus_pdf, mr.x)

    Parameters
    ----------
    func : FitFunc
        Function of params.
    param : np.ndarray[np.float64]
        param values to unwrap.
    as_dict : bool
        Whether to return values as a dictionary or sequence. The default is True.

    Returns
    -------
    out : Sequence[np.ndarray[np.float64]] | dict[str:np.ndarray[np.float64]]
        arrays of ``param0, ... paramN, amps`` .

    """
    out = [param[i::func._nfparam+1] for i in range(func._nfparam+1)]
    if not func._free:
        out[-1] = retrieve_amps(out[-1])
    if as_dict:
        out = dict(zip(func._param_names + ('amps',), out))
    return out


def make_nd_init(func:NdFitFunc, *args:Sequence[np.ndarray[np.float64]], amps:np.ndarray[np.float64]=None, 
                 **kwargs:Sequence[np.ndarray[np.float64]])->np.ndarray[np.float64]:
    """
    Create a parameter array from arrays from sequences of values for each
    parameter type for an ``nd<dist>_...`` distribution func.
    Can specify either as arg, where order is defined by order of underlying func,
    or as kwargs, where names are specified by ``nd<dist>_...`` .

    Parameters
    ----------
    func : NdFitFunc
        Function for which the parameter should be built.
    *args : Sequence[np.ndarray[np.float64]]
        Sequences of value arrays for parameters of func, one value per distribution.
    amps : np.ndarray[np.float64], optional
        Amplitudes per. The default is None.
    **kwargs : Sequence[np.ndarray[np.float64]]
        Sequences of value arrays for parameters of func, one value per distribution.

    Raises
    ------
    TypeError
        Too many or missing parameters specified.
    ValueError
        Mismatched dimensions.

    Returns
    -------
    params : np.ndarray[np.float64]
        nd- parameter array for func.

    """
    args = [[nfunc._invparam(np.atleast_1d(a)) for a in arr] for nfunc, arr in 
            zip(func._ndfuncs, _init_paramargs(func, args, kwargs))]
    if len(args) != len(func._ndfuncs):
        raise TypeError(f"Incorrect number of input param sequences for {func.__name__}")
    if any(len(arr) != len(args[0]) for arr in args[1:]):
        raise ValueError("mismathced number of distributions between inputs")
    if any(any(a.shape != arr[0].shape for a in arr[1:]) for arr in args):
        raise ValueError('mimatched number of dimensions within a param sequence')
    nd = None
    for nfunc, arr in zip(func._ndfuncs, args):
        nd_temp = nfunc._infunc(np.asarray(arr[0]).size)
        if nd is None:
            nd = nd_temp
        elif nd_temp is not None and nd != nd_temp:
            raise ValueError(f"inconsistent number of expected dimensions, {nd} vs {nd_temp}")
    amps = np.ones(len(args[0])) if amps is None else amps
    if not func._free:
        amps, temp = np.empty(amps.shape), amps
        amps[:-1] = rescale_amps(temp)
    params = np.concatenate(list(chain(*zip(*args+[amps.reshape(-1,1)]))))
    if not func._free:
        params, temp = np.empty(params.size-1), params
        params[:] = temp[:-1]
    return params


def make_nd_bounds(func:NdFitFunc, *args:Sequence[tuple[np.ndarray,np.ndarray]], 
                   **kwargs:Sequence[tuple[np.ndarray,np.ndarray]])->np.ndarray[np.float64]:
    """
    Build a bounds Nx2 array for use with :func:`fit_column_mle` or :func:`fit_hist_cdf`
    (and internally passed to minimize_ ) using ``nd<dist>_...`` distributions
    from bounds specified per sub-param.
    Each param should be specified as nested sequence.
    Specify as `[bounds_dist0, bounds_dist1, ..., bounds_distN]``,
    where ``bounds_distn`` is a specificaation of ``[lower, upper]`` where
    ``lower`` and ``upper`` are arrays of appropriate shape for the given parameter.
    (usually either n-size 1D array, or nxn matrix).
    Specify each param either in order of distribution or named accordingly.
    

    Parameters
    ----------
    func : NdFitFunc
        Multi-dimensional multi-distribution function.
    *args : Sequence[tuple[np.ndarray,np.ndarray]]
        Sequences of [lower, upper] bounds arrays.
    **kwargs : Sequence[tuple[np.ndarray,np.ndarray]]
        Sequences of [lower, upper] bounds arrays.

    Raises
    ------
    ValueError
        Incorreclty shaped arrays.

    Returns
    -------
    bounds : np.ndarray[np.float64]
        Nx2 bounds array.

    """
    nd = None
    ndist = None
    barrays = list()
    for nfunc, bound in zip(func._ndfuncs, _init_paramargs(func, args, kwargs)):
        if ndist is None:
            ndist = len(bound)
        elif ndist != len(bound):
            raise ValueError(f"inconsistent number of distributions between bounds {nd} vs {len(ndist)}")
        barray = list()
        for bnd in bound:
            if len(bnd) != 2:
                raise ValueError("Must specify bounds as lower:upper arrays")
            barray.append(np.stack([nfunc._invparam(np.atleast_1d(b)) for b in bnd], axis=1))
            nd_temp = nfunc._infunc(barray[-1].shape[0])
            if nd is None:
                nd = nd_temp
            elif nd_temp is not None and nd != nd_temp:
                raise ValueError("Inconsistent number of dimensions between bounds arrays")
        barrays.append(barray)
    bounds = list(chain(*zip(*barrays+[repeat([[0.0, 1.0]])])))
    if not func._free:
        bounds = bounds[:-1]
    bounds = np.vstack(bounds)
    return bounds

def unwrap_nd_param(func:NdFitFunc, nd:int, param:np.ndarray[np.float64], as_dict:bool=True
                    )->list[list[np.ndarray[np.float64]]]|dict[str:list[np.ndarray[np.float64]]]:
    """
    Separate parameter array of ``n<dist>_...`` type array. 
    Behaves similar to inverse of :func:`make_init`
    
    >>> res = fit_column_mle(data, (columna, columnb), multifit.ngaus_pdf, init)
    >>> mu, sigma, amps = unwrap_nd_param(multifit.ngaus_pdf, res.x)
    
    
    The ``as_dict`` argument will instead return a dictionary with keys of the
    parameter names
    
    >>> resdict = unwrap_nd_param(multifit.ngaus_pdf, res.x, as_dict=True)
    >>> res.keys()
    dict_keys(['mu', 'sigma', 'amps'])


    Parameters
    ----------
    func : FitFunc
        Function of params.
    nd : int
        Number of dimensions expected to be used/were used in optimization.
    param : np.ndarray[np.float64]
        param values to unwrap.
    as_dict : bool
        Whether to return values as a dictionary or sequence. The default is True.

    Returns
    -------
    out : list[list[np.ndarray[np.float64]]] | dict[str:list[np.ndarray[np.float64]]]
        arrays of ``param0, ... paramN, amps`` where each `paramN` is a list of
        value arrays, one element per distribution, or a dictionary with same
        underlying lists, and keys based on parameter names.

    """
    param_ranges = _nparam_size(func, nd)
    naparam = param_ranges[-1] + 1
    ndist = (param.size+int(not func._free)) // naparam
    out = [list() for _ in range(param_ranges.size-1)]
    amps = param[param_ranges[-1]::naparam]
    for i in range(ndist):
        for ipfunc, pb, pe, ot in zip(func._ndfuncs, param_ranges[:-1], param_ranges[1:], out):
            ot.append(ipfunc(param[i*naparam+pb:i*naparam+pe], nd))
    if not func._free:
        amps = retrieve_amps(amps)
    out.append(amps)
    if as_dict:
        out = dict(zip(func._param_names + ('amps',), out))
    return out


def fit_column_mle(data:DataS, column:Column, func:FitFunc, init:np.ndarray, 
                   gate:GateGroup=None, func_kwargs:dict=None, **kwargs)->OptimizeResult:
    """
    Optimize a multi-distribution (``n<dist>_pdf`` type function) based on input
    column using MLE approach.

    Parameters
    ----------
    data : DataS
        Data from which to retrive |Column|.
    column : Column
        |Column| to fit to multi-distribution in ``func``.
    func : FitFunc
        A ``n<dist>_pdf`` like function to optimize MLE vs column.
    init : np.ndarray
        Initial parameter values.
    gate : GateGroup, optional
        |GateGroup| to apply to ``column``. The default is None.
    func_kwargs : dict, optional
        Kwargs handed to func. The default is None.
    **kwargs : Any
        Additional kwargs handed to minimize_ .

    Returns
    -------
    OptimizeResult
        Output of minimize_ .

    """
    getcol = data.get_column if isinstance(data, DataSet) else data.concatenate_column
    col = getcol(column, gate) if hasattr(func, '_nfparam') else np.stack([getcol(col, gate) for col in column], axis=1)
    vfunc = value_mle if func_kwargs is None else partial(value_mle, **func_kwargs)
    return minimize(vfunc, init, args=(col, func), **kwargs)


#: Type hint for functions that can be passed to ``min_func`` of :func:`fit_hist_cdf`.
#: These should have the signature ``min_func(param, bins, hist, func)``
#: and return an optimizeresult_
MinFunc = Callable[[np.ndarray,np.ndarray,np.ndarray,FitFunc],OptimizeResult]

#: Function for use with :func:`fit_hist_pdf`, fits histogram with leastsquare_
lsq_anyfit = partial(least_squares, value_diff)

#: Function for use with :func:`fit_hist_pdf`, fits histogram with minimize_
min_anyfit = partial(minimize, value_lsq)

def fit_hist_pdf(hist:Hist, func:FitFunc, init:np.ndarray, func_kwargs:dict=None, 
                 min_func:MinFunc=lsq_anyfit, **kwargs)->OptimizeResult:
    """
    Perform least-squares fitting of 1D :class:`Hist` against a multi-distribution.
    Should be a ``n<dist>_cdf`` type distribution.

    Parameters
    ----------
    hist : Hist
        Histogram of values to fit.
    func : FitFunc
        CDF Distribution to fit to hist, should be a ``n<dist>_cdf`` type distribution.
    init : np.ndarray
        Initial paramter values for distribution.
    func_kwargs : dict, optional
        Kwargs handed to func. The default is None.
    min_func : MinFunc, optional
        Minimizer function to use, usually either :attr:`lsq_anyfit` to
        use leastsquare_, or :attr:`min_anyfit` to use minimize_.
        The default is lsq_anyfit.
    **kwargs : Any
        Additional kwargs passed to ``min_func``.

    Returns
    -------
    OptimizeResult
        Output of ``min_func``.

    """
    x = hist.bincenters[0] if hasattr(func, '_nfparam') else hist.bincenters
    func = func if func_kwargs is None else partial(func, **func_kwargs)
    return min_func(init, args=(x, hist.pdf, func), **kwargs)


#: Function for use with ``min_func`` argument of :func:`fit_hist_cdf`, fits histogram with leastsquare_
lsq_cdffit = partial(least_squares, value_diff_by_cdf)

#: Function for use with ``min_func`` argument of :func:`fit_hist_cdf`, fits histogram with minimize_
min_cdffit = partial(minimize, value_lsq_by_cdf)


def fit_hist_cdf(hist:Hist, func:FitFunc, init:np.ndarray, func_kwargs:dict=None, 
                 min_func:MinFunc=min_cdffit, **kwargs)->OptimizeResult:
    """
    Perform least-squares fitting of 1D :class:`Hist` against a multi-distribution.
    Should be a ``n<dist>_cdf`` type distribution.

    Parameters
    ----------
    hist : Hist
        Histogram of values to fit.
    func : FitFunc
        CDF Distribution to fit to hist, should be a ``n<dist>_cdf`` type distribution.
    init : np.ndarray
        Initial paramter values for distribution.
    func_kwargs : dict, optional
        Kwargs handed to func. The default is None.
    min_func : Callable[[np.ndarray,np.ndarray,np.ndarray,FitFunc],OptimizeResult]
        Minimizer function to use, usually either :attr:`min_cdffit` to
        use minimize_, or :attr:`min_cdffit` to use leastsquare_.
        The default is min_cdffit.
    **kwargs : Any
        Additional kwargs passed to ``min_func``.

    Returns
    -------
    OptimizeResult
        Output of ``min_func``.

    """
    bins = hist.bins[0] if hasattr(func, '_nfparam') else hist.binlocs
    func = func if func_kwargs is None else partial(func, **func_kwargs)
    return min_func(init, args=(bins, hist.pmf, func), **kwargs)
