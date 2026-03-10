#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created 01/09/2025
# Author: Paul David Harris
"""
.. currentmodule:: fretbursts.immutabledata.gates

Definitions of basic gating definitions and convenience functions.


"""
from typing import Any
from collections.abc import Callable
from itertools import product
from numbers import Real

import numpy as np

from .utils import tupledict
from .immutabledata import register_PyCode
from .tables import (
    Column, GateDefinition, GateDef, MappedGateDef, Gate, MappedGate, GateGroup, 
    _TT_ft, _TT_tf, TT_subtract,
    GD_intersect, GD_equal, GD_superset, GD_subset
                     )


def _linear_compute(columns:tuple[np.ndarray[np.floating],...],
                    vec:np.ndarray[np.floating])->np.ndarray[np.floating]:
    """Compute value of dotproduct of columns and vec"""
    return np.sum(np.array(columns).T*vec, axis=1)


def linear_gt_gate(*columns:np.ndarray[np.floating], vec:np.ndarray[np.double], m:float)->np.ndarray[np.bool_]:
    r"""
    Gate function for :attr:`LIN_GT_gate`. :math:`V \cdot C > m`

    Parameters
    ----------
    *columns : np.ndarray[np.floating]
        arrays of each column of gate, number of columns = n, each element becomes
        index *i* of *C*.
    vec : np.ndarray[np.double]
        size n 1D magnitude 1 vector ( *V* ).
    m : float
        threshold, true if :math:`V \cdot C > m`.

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return _linear_compute(columns, vec) > m


register_PyCode(linear_gt_gate)


def linear_geq_gate(*columns:np.ndarray[np.floating], vec:np.ndarray[np.double], m:float)->np.ndarray[np.bool_]:
    r"""
    Gate function for :attr:`LIN_GTE_gate`. :math:`V \cdot C >= m`

    Parameters
    ----------
    *columns : np.ndarray[np.floating]
        arrays of each column of gate, number of columns = n, each element becomes
        index *i* of *C*.
    vec : np.ndarray[np.double]
        size n 1D magnitude 1 vector ( *V* ).
    m : float
        threshold, true if :math:`V \cdot C >= m`.

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return _linear_compute(columns, vec) >= m


register_PyCode(linear_geq_gate)


def _regularize_linear_gate(params:dict[str:Any], colorder:tuple[int,...])->dict[str:Any]:
    """Ensure that columns in correct order, and vec is unit vector"""
    if 'vec' not in params:
        raise ValueError("must specify vec for linear_gate")
    vec = np.atleast_1d(params['vec'])
    if vec.ndim != 1:
        raise ValueError(f"vec must be 1d array, (input is {vec.ndim}")
    elif vec.shape[0] != len(colorder):
        raise ValueError(f"vec must be same size as columns, got {vec.shape[0]}, expected {len(colorder)}")
    m = float(params.get('m', 0.0))
    colorder = np.array(colorder, dtype=np.int64)
    vec = vec[colorder]
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    m /= norm
    if (err:=np.argwhere(vec == 0.0)).shape[0]:
        raise ValueError(f"element {err[0,0]} of vec to small, is 0.0 after normalization")
    if vec[0] < 0.0:
        vec, m = -vec, -m
    return dict(vec=vec, m=m)


register_PyCode(_regularize_linear_gate)

#: GateDef for :func:`linear_gt_gate` discouraged from use, use :attr:`LIN_GEQ_gate` instead.
LIN_GT_gate = GateDef(linear_gt_gate, tupledict(('vec', np.ndarray), ('m', float)),
                   sortcol=True, regularize=_regularize_linear_gate)

#: GateDef for :func:`linear_geq_gate`
LIN_GEQ_gate = GateDef(linear_geq_gate, tupledict(('vec', np.ndarray), ('m', float)),
                   sortcol=True, regularize=_regularize_linear_gate)


def _comp_linear(gateA:Gate, gateB:Gate)->int:
    """Comparison function for 2 linear type gates"""
    if gateA.columns != gateB.columns:
        return 0b1111
    if np.any(gateA.params['vec'] != gateB.params['vec']):
        return 0b1111
    if gateA.params['m'] < gateB.params['m']:
        return 0b1011
    elif gateA.params['m'] > gateB.params['m']:
        return 0b1101
    if gateA.gatedef == LIN_GT_gate and gateB.gatedef == LIN_GEQ_gate:
        return 0b1101
    elif gateA.gatedef == LIN_GEQ_gate and gateB.gatedef == LIN_GEQ_gate:
        return 0b1011
    return 0b1001


GateDefinition.set_gate_comparison(LIN_GT_gate, LIN_GT_gate, _comp_linear)
GateDefinition.set_gate_comparison(LIN_GEQ_gate, LIN_GEQ_gate, _comp_linear)
GateDefinition.set_gate_comparison(LIN_GT_gate, LIN_GEQ_gate, _comp_linear)


def _ellipsoid_compute(columns:tuple[np.ndarray[np.floating]], transform:np.ndarray[np.floating],
                       center:np.ndarray[np.floating])->np.ndarray[np.floating]:
    """Compute values of transformed column vectors"""
    return np.sum((transform @ np.array(columns)-center[:, np.newaxis])**2, axis=0)


def ellipsoid_lt_gate(*columns:np.ndarray[np.floating], transform:np.ndarray[np.floating],
                      center:np.ndarray[np.floating])->np.ndarray[np.bool_]:
    r"""
    Gate function for :attr:`ELLIPSOID_LTE_gate`
    Creates an n-D ellipsoid with an open (less than) boarder.

    Parameters
    ----------
    \*columns : Column
        arrays of each column of gate, number of columns = n, each element becomes
        index *i* of *C*.
    transform : np.ndarray[np.double]
        The :math:`\mathbf{T}` in :math:`\left\|{\mathbf{T}\vec{x}-\vec{c}}\right\| \lt 1`.
    center : np.ndarray[np.double]
        The :math:`\vec{c}` in :math:`\left\|{\mathbf{T}\vec{x}-\vec{c}}\right\| \lt 1`.

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return _ellipsoid_compute(columns, transform, center) < 1.0


register_PyCode(ellipsoid_lt_gate)

                
def ellipsoid_leq_gate(*columns:Column, transform:np.ndarray[np.double], center:np.ndarray[np.double])->np.ndarray[np.bool_]:
    r"""
    Gate function for :attr:`ELLIPSOID_LEQ_gate`
    Creates an n-D ellipsoid with a closed (less than or equal to) 
    boarder.

    Parameters
    ----------
    *columns : Column
        arrays of each column of gate, number of columns = n, each element becomes
        index *i* of *C*.
    transform : np.ndarray[np.double]
        The :math:`\mathbf{T}` in :math:`\left\|{\mathbf{T}\vec{x}-\vec{c}}\right\| \leq 1`.
    center : np.ndarray[np.double]
        The :math:`\vec{c}` in :math:`\left\|{\mathbf{T}\vec{x}-\vec{c}}\right\| \leq 1`.

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return _ellipsoid_compute(columns, transform, center) <= 1.0


register_PyCode(ellipsoid_leq_gate)


def _reg_upper(r:np.ndarray[np.double])->np.ndarray[np.double]:
    """For ellipse rotation matrix, generate upper triangular matrix of r"""
    rn = np.zeros(r.shape)
    for i in range(r.shape[0]):
        rn[i,i] = np.sqrt(np.sum(r[:,i]**2)-np.sum(rn[:i,i]**2))
        for j in range(i, r.shape[1]):
            rn[i,j] = (np.sum(r[:,i]*r[:,j])-np.sum(rn[:i,i]*rn[:i,j]))/rn[i,i]
    return rn


def _regularize_ellipsoide_gate(params:dict[str:np.ndarray[np.double]],
                               colorder:tuple[int,...])->np.ndarray[np.bool_]:
    """
    Enure transform is upper triangular matrix and re-order transform and
    center according to colorder
    """
    center = params.pop('center', np.zeros(len(colorder)))
    s = colorder.shape[0]
    if center.ndim != 1 or center.shape[0] != s:
        raise ValueError(f"center must be 1-D array of shape {colorder.shape}, has shape {center.shape}")
    transform = params.pop('transform', np.eye(len(colorder)))
    if transform.ndim != 2 or any(sh != s for sh in transform.shape):
        raise ValueError("transform must be ")
    transform = transform[colorder]
    if params:
        raise ValueError(f"unrecognized param(s) for elipsoid_gate: {tuple(params.keys())}")
    center = center[colorder]
    if any(transform[i,j] != 0.0 for i, j in product(range(s), range(s)) if j > i):
        transform = _reg_upper(transform)
    return dict(transform=transform, center=center)


register_PyCode(_regularize_ellipsoide_gate)


#: GateDef for :func:`ellipspoid_lt_gate`
ELLIPSOID_LT_gate = GateDef(ellipsoid_lt_gate, tupledict(('transform', np.ndarray), ('center', np.ndarray)), 
                             sortcol=True, regularize=_regularize_ellipsoide_gate)

#: GateDef for :func:`ellipspoid_leq_gate, discouraged from use`
ELLIPSOID_LEQ_gate = GateDef(ellipsoid_leq_gate, tupledict(('transform', np.ndarray), ('center', np.ndarray)), 
                             sortcol=True, regularize=_regularize_ellipsoide_gate)


def _isin_nan(elements:np.ndarray, test_elements:np.ndarray, **kwargs)->np.ndarray[np.bool_]:
    """
    Nan-matching isin function. Wraps np.isin

    Parameters
    ----------
    elements : np.ndarray
        Input array.
    test_elements : np.ndarray
        The values against which to test each value of element. 
    **kwargs : TYPE
        Passed to np.isin.

    Returns
    -------
    np.ndarray[np.bool\_]
        Boolean mask of whether element in elements is in test_elements.

    """
    if np.issubdtype(test_elements.dtype, np.floating):
        isnan = np.isnan(test_elements)
        if np.all(isnan):
            return np.isnan(elements)
        elif np.any(isnan):
            return np.isnan(elements) | np.isin(elements, test_elements, **kwargs)
    return np.isin(elements, test_elements, **kwargs)


def isin_gate(column:np.ndarray, *, inset:np.ndarray)->np.ndarray[np.bool_]:
    """
    Gate function for :attr:`ISIN_gate`

    Parameters
    ----------
    column : np.ndarray
        array mask.
    inset : np.ndarray, optional
        Set of values for which gate evaluates True. The default is frozenset().

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return _isin_nan(column, inset)


def _regularize_isin(params:dict, cols:tuple[Column])->dict:
    """Ensure params['isin'] matches type fo col"""
    inset = params['inset']
    dtype = cols[0]._get_coldef().dtype
    params['inset'] = inset.astype(dtype).reshape(-1)
    return params


def _comp_isin_isin(gateA:Gate, gateB:Gate)->int:
    """Comparison function for comparing two ISIN_gate gates"""
    if gateA.columns[0] != gateB.columns[0]:
        return GD_intersect
    ainb = _isin_nan(gateA.params['inset'], gateB.params['inset'])
    bina = _isin_nan(gateB.params['inset'], gateA.params['inset'])
    return np.any(ainb)<<3 | (not np.all(bina))<<2 | (not np.all(ainb))<<1 | 0b0001


def _comp_isin_lin(gateIN:Gate, gateLN:Gate, comp:Callable[[np.ndarray, Real],np.ndarray[np.bool_]])->int:
    """Comparison function for comparing ISIN_gate with LINEAR_XX_gate gates"""
    if gateIN.columns[0] in gateLN.columns:
        inset = gateIN.params['inset']
        if len(gateLN.columns) > 1:
            if np.all(np.isnan(inset)):
                return 0b0110
            return GD_intersect
        ingate = inset > gateLN.params['m']
        return np.any(ingate)<<3 | 1<<2 | (not np.all(ingate))<<1 | 1
    return GD_intersect


def _comp_isin_gt(gateIN:Gate, gateGT:Gate)->int:
    """Comparison function for comparing ISIN_gate with LINEAR_GT_gate gates"""
    return _comp_isin_lin(gateIN, gateGT, lambda inset, m: inset > m)


def _comp_isin_geq(gateIN:Gate, gateGEQ:Gate)->int:
    """Comparison function for comparing ISIN_gate with LINEAR_GEQ_gate gates"""
    return _comp_isin_lin(gateIN, gateGEQ, lambda inset, m: inset >= m)


register_PyCode(isin_gate)

#: GateDef for :func:`isin_gate` function
ISIN_gate = GateDef(isin_gate, tupledict(('inset', np.ndarray)), np.array([1,1]))
GateDefinition.set_gate_comparison(ISIN_gate, ISIN_gate, _comp_isin_isin)
GateDefinition.set_gate_comparison(ISIN_gate, LIN_GT_gate, _comp_isin_gt)
GateDefinition.set_gate_comparison(ISIN_gate, LIN_GEQ_gate, _comp_isin_geq)


def percentile_geq_gate(col:np.ndarray[np.number], *, percentile:float):
    r"""
    Gate function that takes all rows in with values greater than or equal to
    the `percentile` percentile.
    Gate function for :attr:`PERCENTILE_GT_gate`

    Parameters
    ----------
    col : np.ndarray[np.number]
        array mask.
    percentile : float
        percecntile including and above which to have an element of col be True.

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return col >= np.percentile(col, percentile)


register_PyCode(percentile_geq_gate)


def percentile_gt_gate(col:np.ndarray[np.number], *, percentile:float):
    r"""
    **Discouraged from use** use :func:`percentile_geq_gate` instead.
    
    Gate function that takes all rows in with values greater than to
    the `percentile` percentile.
    Gate function for :attr:`PERCENTILE_GT_gate`

    Parameters
    ----------
    col : np.ndarray[np.number]
        array mask.
    percentile : float
        percecntile bove which an element of col will be True.

    Returns
    -------
    np.ndarray[np.bool\_]
        gate mask.

    """
    return col > np.percentile(col, percentile)


register_PyCode(percentile_gt_gate)


def _regularize_percentile(param:dict[str:float])->dict:
    """regularize function for percentile_XXX_gate GateDefs"""
    if 'percentile' not in param:
        param['percentile'] = 90.0
    if param['percentile'] <= 0.0 or param['percentile'] >= 100.0:
        raise ValueError("'percentile' must be in range of (0.0, 100.0)")
    return param


register_PyCode(_regularize_percentile)

#: GateDef for :func:`percentile_geq_gate`
PERCENTILE_GEQ_gate = GateDef(percentile_geq_gate, tupledict(('percentile',float)), atomic=False, regularize=_regularize_percentile)

#: GateDef for :func:`percentile_gt_gate`, discouraged from use
PERCENTILE_GT_gate = GateDef(percentile_gt_gate, tupledict(('percentile',float)), atomic=False, regularize=_regularize_percentile)


def shift_mask(mask:np.ndarray[np.bool_], offset:int=1, fill:bool=False)->np.ndarray[np.bool_]:
    r"""
    Shift the values of a boolean mask by offset.

    Parameters
    ----------
    mask : np.ndarray[np.bool\_]
        boolean mask.
    offset : int, optional
        Number of indexes to offset mask. The default is 1.
    fill : bool, optional
        Value of mask in first or last offset indexes. The default is False.

    Returns
    -------
    np.ndarray[np.bool\_]
        Shifted mask.

    """
    out = np.empty(mask.size, dtype=np.bool_)
    if offset < 0:
        oslc = slice(None, offset)
        islc = slice(-offset, None)
        fslc = slice(offset,None)
    else:
        oslc = slice(None, -offset)
        islc = slice(offset, None)
        fslc = slice(None,offset)
    out = np.empty(mask.shape, dtype=np.bool_)
    out[oslc] = mask[islc]
    out[fslc] = fill
    return out


def _validate_shift_mask(params):
    """Check function for shift-type MappedGate"""
    if params.offset == 0:
        raise ValueError("shift_mask cannot have offset == 0 (this results in same result as mask_gate)")


register_PyCode(shift_mask)
register_PyCode(_validate_shift_mask)

#: MappedGateDef for :func:`shift_mask` shifts gate by offset
SHIFT_mapgate = MappedGateDef(shift_mask, tupledict(('offset', int), ('fill', bool)), _validate_shift_mask)


###############################################################################
################### Convenience functions for making gates  ###################
###############################################################################
_nan_exclude = np.array([np.nan])
_nan_exclude.setflags(write=False)

def make_exclude_nan(column:Column)->GateGroup:
    """
    Create a gate that excludes all rows that have NAN values for a given column.

    Parameters
    ----------
    column : Column
        Column of gate to be created.

    Raises
    ------
    TypeError
        Column cannot have NAN (is not a floating point column).

    Returns
    -------
    GateGroup
        Gate excluding rows with NANs of given column.

    """
    if not np.issubdtype(column._get_coldef().dtype, np.floating):
        raise TypeError("Column excluding nan must ")
    return GateGroup(_TT_tf, Gate(ISIN_gate, (column, ), _nan_exclude))


_TT_andnn = np.array([[True, False,],[False,False]])
_TT_andnn.setflags(write=False)


def make_gt_gate(column:Column, mn:float, exclude_nan:bool=True, outside_expand:bool=False)->GateGroup:
    """
    Create a gate for all values of column greater than mn.
    This gate is discourage from use, use :func:`make_geq_gate` instead.

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mn : float
        minnimum value of gate (exclusive, ie greater than).
    exclude_nan : bool, optional
        Whether to exclude NAN values from gate.
        The default is True
    outside_expand : bool, optional
        Whether gate should incldue rows outsid of parent_gate. 
        **Only for non-atomic columns**, ignored if column is atomic.
        The default is False.

    Returns
    -------
    GateGroup
        Gate representing all rows with column value greater than mn.

    """
    if column.atomic:
        gate = Gate(LIN_GT_gate, (column, ), dict(vec=np.array([1.0]), m=float(mn)))
    else:
        gate =Gate(LIN_GT_gate, (column, ), dict(vec=np.array([1.0]), m=float(mn)), 
                   expand=outside_expand)
    title = fr'{column.name()} \gt {mn}'
    if np.issubdtype(column._get_coldef().dtype, np.floating) and not exclude_nan:
        nexcl = Gate(ISIN_gate, (column, ), {'inset':_nan_exclude})
        out = GateGroup(TT_subtract, gate, nexcl, title=title)
    else:
        out = GateGroup(_TT_ft, gate, title=title)
    return out

    
def make_geq_gate(column:Column, mn:float, exclude_nan:bool=True, outside_expand:bool=False)->GateGroup:
    """
    Create a gate for all values of column greater than or equal to mn.

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mn : float
        minnimum value of gate (inclusive, ie greater than or equal to).
    exclude_nan : bool, optional
        Whether to exclude NAN values from gate.
        The default is True
    outside_expand : bool, optional
        Whether gate should incldue rows outsid of parent_gate. 
        **Only for non-atomic columns**, ignored if column is atomic.
        The default is False.

    Returns
    -------
    GateGroup
        Gate representing all rows with column value greater than or equal to mn.

    """
    if column.atomic:
        gate = Gate(LIN_GEQ_gate, (column, ), dict(vec=np.array([1.0]), m=float(mn)))
    else:
        gate =Gate(LIN_GEQ_gate, (column, ), dict(vec=np.array([1.0]), m=float(mn)), 
                   expand=outside_expand)
    title = fr'{column.name()} \geq {mn}'
    if np.issubdtype(column._get_coldef().dtype, np.floating) and not exclude_nan:
        nexcl = Gate(ISIN_gate, column, {'inset':_nan_exclude})
        out = GateGroup(TT_subtract, gate, nexcl, title=title)
    else:
        out = GateGroup(_TT_ft, gate, title=title)
    return out


def make_lt_gate(column:Column, mx:float, exclude_nan:bool=True, outside_expand:bool=False)->GateGroup:
    """
    Create a gate for all values of column less than mx.

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mx : float
        maximum value of column (exclusive, ie less than).
    exclude_nan : bool, optional
        Whether to exclude NAN values from gate.
        The default is True
    outside_expand : bool, optional
        Whether gate should incldue rows outsid of parent_gate. 
        **Only for non-atomic columns**, ignored if column is atomic.
        The default is False.

    Returns
    -------
    GateGroup
        Gate representing all rows with column value less than mx.

    """
    if column.atomic:
        gate = Gate(LIN_GEQ_gate, column, dict(vec=np.array([1.0]), m=float(mx)))
    else:
        gate = Gate(LIN_GEQ_gate, column, dict(vec=np.array([1.0]), m=float(mx)), 
                    expand= not outside_expand)
    title = fr'{column.name()} \lt {mx}'
    if np.issubdtype(column._get_coldef().dtype, np.floating) and exclude_nan:
        nexcl = Gate(ISIN_gate, (column, ), {'inset':_nan_exclude})
        out = GateGroup(_TT_andnn, gate, nexcl, title=title)
    else:
        out = GateGroup(_TT_tf, gate, title=title)
    return out


def make_leq_gate(column:Column, mx:float, exclude_nan:bool=True, outside_expand:bool=False)->GateGroup:
    """
    Create a gate for all values of column less than or equal to mx.
    This gate is discrouraged from use, use :func:`make_lt_gate` instead.

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mx : float
        maximum value of column (inclusive, ie less than or equal to).
    exclude_nan : bool, optional
        Whether to exclude NAN values from gate.
        The default is True
    outside_expand : bool, optional
        Whether gate should incldue rows outsid of parent_gate. 
        **Only for non-atomic columns**, ignored if column is atomic.
        The default is False.

    Returns
    -------
    GateGroup
        Gate representing all rows with column value less than or equal to mx.

    """
    if column.atomic:
        gate = Gate(LIN_GT_gate, column, dict(vec=np.array([1.0]), m=float(mx)))
    else:
        gate = Gate(LIN_GT_gate, column, dict(vec=np.array([1.0]), m=float(mx)), 
                    expand= not outside_expand)
    title = fr'{column.name()} \leq {mx}'

    if np.issubdtype(column._get_coldef().dtype, np.floating) and exclude_nan:
        nexcl = Gate(ISIN_gate, (column, ), {'inset':_nan_exclude})
        out = GateGroup(_TT_andnn, gate, nexcl, title=title)
    else:
        out = GateGroup(_TT_tf, gate, title=title)
    return out


def _rotation_matrix(theta:float)->np.ndarray[np.double]:
    """Crate 2x2 (2D) rotation matrix for theta (in radians)"""
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])


def _make_ellipsoid_gate(cx:float=0, cy:float=0, dx:float=1.0, dy:float=1.0, 
                         theta:float=0.0, radians:bool=False)->dict:
    """Create params for ellipse gate based on center/w/h/rot"""
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("w and h must be positive")
    center = -np.array([cx, cy], dtype=np.double)
    stretch = np.array([dx, dy])[:,np.newaxis]
    theta = theta if radians else np.deg2rad(theta)
    rmat = _rotation_matrix(theta)
    transform = 2.0/stretch * rmat.T # 0.5/stretch because convert from radius to diameter
    return dict(transform=transform, center=center)


def make_ellipsoid_gate(colx:Column, coly:Column,
                        cx:float=0, cy:float=0,
                        dx:float=1.0, dy:float=1.0,
                        theta:float=0.0, radians:bool=False)->GateGroup:
    """
    Create a gate based on colx and coly of an ellipse centered at (``cx, cy``)
    with widths and height ``w, y``, and rotated by the angle ``theta``. All
    points inside ellipse included in gate.

    Parameters
    ----------
    colx : Column
        X-axis column of gate.
    coly : Column
        Y-axis column of gate.
    cx : float, optional
        Center of ellipse in x. The default is 0.
    cy : float, optional
        Center of ellipse in x. The default is 0.
    dx : float, optional
        width (x-axis diameter) of ellipse. The default is 1.0.
    dy : float, optional
        height (y-axis diameter) of ellipse. The default is 1.0.
    theta : float, optional
        rotation angle read as degrees by default, if ``radians=True``, read as
        radians. Angle by which ellips is rotated. Note that rotaion of 90 degrees
        will result in effective swapping of cx/cy/w/h parameters. Additionally
        rotation outside of [-45, 45] degrees could be equivalently achieved
        by swapping x and y axes and using an angle in the above range, therefore
        it is encouraged to use only angles betwee [-45, 45] degrees.
        The default is 0.0.
    radians : bool, optional
        If ``True`` read theta in radians, if ``False`` read in degrees.
        The default is False.

    Returns
    -------
    GateGroup
        GateGroup representing All values inside of ellipse boundry, open boundary
        (ie points on boudary not included).

    """
    params = _make_ellipsoid_gate(cx, cy, dx, dy, theta, radians)
    return GateGroup(_TT_ft, Gate(ELLIPSOID_LT_gate, (colx, coly), params))


def make_ellipsoid_inclusive_gate(colx:Column, coly:Column, 
                                  cx:float=0, cy:float=0, 
                                  w:float=1.0, h:float=1.0, 
                                  theta:float=0.0, radians:bool=False)->GateGroup:
    """
    Create a gate based on colx and coly of an ellipse centered at (``cx, cy``)
    with widths and height ``w, y``, and rotated by the angle ``theta``. All
    points inside and including the ellipse included in gate.

    Parameters
    ----------
    colx : Column
        X-axis column of gate.
    coly : Column
        Y-axis column of gate.
    cx : float, optional
        Center of ellipse in x. The default is 0.
    cy : float, optional
        Center of ellipse in x. The default is 0.
    w : float, optional
        width (x-axis diameter) of ellipse. The default is 1.0.
    h : float, optional
        height (y-axis diameter) of ellipse. The default is 1.0.
    theta : float, optional
        rotation angle read as degrees by default, if ``radians=True``, read as
        radians. Angle by which ellips is rotated. Note that rotaion of 90 degrees
        will result in effective swapping of cx/cy/w/h parameters. Additionally
        rotation outside of [-45, 45] degrees could be equivalently achieved
        by swapping x and y axes and using an angle in the above range, therefore
        it is encouraged to use only angles betwee [-45, 45] degrees.
        The default is 0.0.
    radians : bool, optional
        If ``True`` read theta in radians, if ``False`` read in degrees.
        The default is False.

    Returns
    -------
    GateGroup
        GateGroup representing All values inside of ellipse boundry, closed boundary
        (ie points on boudary included).

    """
    params = _make_ellipsoid_gate(cx, cy, w, h, theta, radians)
    return GateGroup.as_gategroup(Gate(ELLIPSOID_LEQ_gate, (colx, coly), params))


def make_inv_ellipsoid_gate(colx:Column, coly:Column, cx:float=0, cy:float=0, w:float=1.0, h:float=1.0, rot=0.0, radians:bool=False)->GateGroup:
    """
    Create a gate based on colx and coly of an ellipse centered at (``cx, cy``)
    with widths and height ``w, y``, and rotated by the angle ``theta``. All
    points outside ellipse included in gate.

    Parameters
    ----------
    colx : Column
        X-axis column of gate.
    coly : Column
        Y-axis column of gate.
    cx : float, optional
        Center of ellipse in x. The default is 0.
    cy : float, optional
        Center of ellipse in x. The default is 0.
    w : float, optional
        width (x-axis diameter) of ellipse. The default is 1.0.
    h : float, optional
        height (y-axis diameter) of ellipse. The default is 1.0.
    theta : float, optional
        rotation angle read as degrees by default, if ``radians=True``, read as
        radians. Angle by which ellips is rotated. Note that rotaion of 90 degrees
        will result in effective swapping of cx/cy/w/h parameters. Additionally
        rotation outside of [-45, 45] degrees could be equivalently achieved
        by swapping x and y axes and using an angle in the above range, therefore
        it is encouraged to use only angles betwee [-45, 45] degrees.
        The default is 0.0.
    radians : bool, optional
        If ``True`` read theta in radians, if ``False`` read in degrees.
        The default is False.

    Returns
    -------
    GateGroup
        GateGroup representing All values outside of ellipse boundry, open boundary
        (ie points on boudary not included).

    """
    params = _make_ellipsoid_gate(cx, cy, w, h, rot, radians)
    return ~Gate(ELLIPSOID_LEQ_gate, (colx, coly), params)


def make_inv_ellipsoid_inclusive_gate(colx:Column, coly:Column, cx:float=0, cy:float=0, w:float=1.0, h:float=1.0, rot=0.0, radians:bool=False)->GateGroup:
    """
    Create a gate based on colx and coly of an ellipse centered at (``cx, cy``)
    with widths and height ``w, y``, and rotated by the angle ``theta``. All
    points outside and including ellipse included in gate.

    Parameters
    ----------
    colx : Column
        X-axis column of gate.
    coly : Column
        Y-axis column of gate.
    cx : float, optional
        Center of ellipse in x. The default is 0.
    cy : float, optional
        Center of ellipse in x. The default is 0.
    w : float, optional
        width (x-axis diameter) of ellipse. The default is 1.0.
    h : float, optional
        height (y-axis diameter) of ellipse. The default is 1.0.
    theta : float, optional
        rotation angle read as degrees by default, if ``radians=True``, read as
        radians. Angle by which ellips is rotated. Note that rotaion of 90 degrees
        will result in effective swapping of cx/cy/w/h parameters. Additionally
        rotation outside of [-45, 45] degrees could be equivalently achieved
        by swapping x and y axes and using an angle in the above range, therefore
        it is encouraged to use only angles betwee [-45, 45] degrees.
        The default is 0.0.
    radians : bool, optional
        If ``True`` read theta in radians, if ``False`` read in degrees.
        The default is False.

    Returns
    -------
    GateGroup
        GateGroup representing All values outside of ellipse boundry, closed boundary
        (ie points on boudary are included).

    """
    params = _make_ellipsoid_gate(cx, cy, w, h, rot, radians)
    return ~Gate(ELLIPSOID_LT_gate, (colx, coly), params)


def make_upper_inclusive_percentile_gate(column:Column, up:float)->GateGroup:
    r"""
    Create a gate for column taking only values in the `up`\ :sup:`th` percentile
    or greater. (Greater than or equal to gate, ie inclusive).

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    up : float
        upper-percentile, ie the percentile that values must be greater than or
        equal to.

    Returns
    -------
    GateGroup
        Gate for all rows in a table with a column value in the `up`\ :sup:`th` 
        percentile or greater.

    """
    return GateGroup.as_gategroup(Gate(PERCENTILE_GEQ_gate, column, dict(percentile=up)))


def make_upper_exclusive_percentile_gate(column:Column, up:float)->GateGroup:
    r"""
    Create a gate for column taking only values larger than the `up`\ :sup:`th` 
    percentile. (Greater than gate, ie exclusive).

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    up : float
        upper-percentile, ie the percentile that values must be greater than.

    Returns
    -------
    GateGroup
        Gate for all rows in a table with a column value greater than the 
        `up`\ :sup:`th` percentile.

    """
    return GateGroup.as_gategroup(Gate(PERCENTILE_GT_gate, column, dict(percentile=up)))


def make_lower_inclusive_percentile_gate(column:Column, lp:float)->GateGroup:
    r"""
    Create a gate for column taking only values in the `lp`\ :sup:`th` percentile
    or smaller. (Less than or equal to gate, ie inclusive).

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    lp : float
        lower-percentile, ie the percentile that values must be less than or
        equal to.

    Returns
    -------
    GateGroup
        Gate for all rows in a table with a column value in the `up`\ :sup:`th` 
        percentile or greater.

    """
    return ~Gate(PERCENTILE_GT_gate, (column, ), dict(percentile=lp))


def make_lower_exclusive_percentile_gate(column:Column, low:float)->GateGroup:
    r"""
    Create a gate for column taking only values larger than the `up`\ :sup:`th` 
    percentile. (Greater than gate, ie exclusive).

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    up : float
        upper-percentile, ie the percentile that values must be greater than.

    Returns
    -------
    GateGroup
        Gate for all rows in a table with a column value greater than the 
        `up`\ :sup:`th` percentile.

    """
    return ~Gate(PERCENTILE_GEQ_gate, (column, ), dict(percentile=low))