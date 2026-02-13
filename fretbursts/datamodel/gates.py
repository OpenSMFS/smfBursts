#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created 01/09/2025
# Author: Paul David Harris
"""
.. currentmodule:: fretbursts.immutabledata.gates

Definitions of basic gating definitions and convenience functions.


"""
from typing import Any
from itertools import product

import numpy as np

from .utils import tupledict
from .immutabledata import register_PyCode
from .tables import Column, GateDef, Gate, GateGroup, _TT_ft, _TT_tf
from .tables import GD_intersect, GD_equal, GD_superset, GD_subset


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
    np.ndarray[np.bool_]
        gate mask.

    """
    return _linear_compute(columns, vec) > m


register_PyCode(linear_gt_gate)


def linear_gte_gate(*columns:np.ndarray[np.floating], vec:np.ndarray[np.double], m:float)->np.ndarray[np.bool_]:
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
    np.ndarray[np.bool_]
        gate mask.

    """
    return _linear_compute(columns, vec) >= m


register_PyCode(linear_gte_gate)


def _normalize_linear_gate(params:dict[str, Any], colorder:tuple[int,...]):
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


register_PyCode(_normalize_linear_gate)


LIN_GT_gate = GateDef(linear_gt_gate, tupledict(('vec', np.ndarray), ('m', float)),
                   sortcol=True, normalize=_normalize_linear_gate)


LIN_GTE_gate = GateDef(linear_gte_gate, tupledict(('vec', np.ndarray), ('m', float)),
                   sortcol=True, normalize=_normalize_linear_gate)


def _comp_linear(gateA:Gate, gateB:Gate)->int:
    """Comparison function for 2 linear type gates"""
    if gateA.columns != gateB.columns:
        return GD_intersect
    if np.any(gateA.params['vec'] != gateB.params['vec']):
        return GD_intersect
    if gateA.params['m'] < gateB.params['m']:
        return GD_superset
    elif gateA.params['m'] > gateB.params['m']:
        return GD_subset
    if gateA.gatedef == LIN_GT_gate and gateB.gatedef == LIN_GTE_gate:
        return GD_subset
    elif gateA.gatedef == LIN_GTE_gate and gateB.gatedef == LIN_GT_gate:
        return GD_superset
    return GD_equal


GateDef.set_gate_comparison(LIN_GT_gate, LIN_GT_gate, _comp_linear)
GateDef.set_gate_comparison(LIN_GTE_gate, LIN_GTE_gate, _comp_linear)
GateDef.set_gate_comparison(LIN_GT_gate, LIN_GTE_gate, _comp_linear)


def _ellipsoid_compute(columns:tuple[np.ndarray[np.floating]], transform:np.ndarray[np.floating],
                       center:np.ndarray[np.floating])->np.ndarray[np.floating]:
    """Compute values of transformed column vectors"""
    return np.sum((transform @ np.array(columns)-center[:, np.newaxis])**2, axis=0)


def ellipsoid_lt_gate(*columns:np.ndarray[np.floating], transform:np.ndarray[np.floating],
                      center:np.ndarray[np.floating])->np.ndarray[np.bool_]:
    """
    Gate function for :attr:`ELLIPSOID_LTE_gate`
    Creates an n-D ellipsoid with an open (less than) boarder.

    Parameters
    ----------
    *columns : Column
        arrays of each column of gate, number of columns = n, each element becomes
        index *i* of *C*.
    transform : np.ndarray[np.double]
        transformation matrix to apply to vector of location in columns.
    center : np.ndarray[np.double]
        Center vector, ``magnitude(transform@(loc-center)) < 1.0``.

    Returns
    -------
    np.ndarray[np.bool_]
        gate mask.

    """
    return _ellipsoid_compute(columns, transform, center) < 1.0


register_PyCode(ellipsoid_lt_gate)

                
def ellipsoid_lte_gate(*columns:Column, transform:np.ndarray[np.double], center:np.ndarray[np.double])->np.ndarray[np.bool_]:
    """
    Gate function for :attr:`ELLIPSOID_LTE_gate`
    Creates an n-D ellipsoid with a closed (less than or equal to) 
    boarder.

    Parameters
    ----------
    *columns : Column
        arrays of each column of gate, number of columns = n, each element becomes
        index *i* of *C*.
    transform : np.ndarray[np.double]
        DESCRIPTION.
    center : np.ndarray[np.double]
        DESCRIPTION.

    Returns
    -------
    np.ndarray[np.bool_]
        gate mask.

    """
    return _ellipsoid_compute(columns, transform, center) <= 1.0


register_PyCode(ellipsoid_lte_gate)


def _norm_upper(r:np.ndarray[np.double])->np.ndarray[np.double]:
    """For ellipse rotation matrix, generate upper triangular matrix of r"""
    rn = np.zeros(r.shape)
    for i in range(r.shape[0]):
        rn[i,i] = np.sqrt(np.sum(r[:,i]**2)-np.sum(rn[:i,i]**2))
        for j in range(i, r.shape[1]):
            rn[i,j] = (np.sum(r[:,i]*r[:,j])-np.sum(rn[:i,i]*rn[:i,j]))/rn[i,i]
    return rn


def _normalize_ellipsoide_gate(params:dict[str, np.ndarray[np.double]],
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
        transform = _norm_upper(transform)
    return dict(transform=transform, center=center)


register_PyCode(_normalize_ellipsoide_gate)


ELLIPSOID_LT_gate = GateDef(ellipsoid_lt_gate, tupledict(('transform', np.ndarray), ('center', np.ndarray)), 
                             sortcol=True, normalize=_normalize_ellipsoide_gate)

ELLIPSOID_LTE_gate = GateDef(ellipsoid_lte_gate, tupledict(('transform', np.ndarray), ('center', np.ndarray)), 
                             sortcol=True, normalize=_normalize_ellipsoide_gate)


def isin_gate(column:np.ndarray[np.integer], *, inset:np.ndarray[np.integer])->np.ndarray[np.bool_]:
    """
    Gate function for :attr:`ISIN_gate`

    Parameters
    ----------
    column : np.ndarray[np.integer]
        array mask.
    inset : np.ndarray[np.integer], optional
        Set of values for which gate evaluates True. The default is frozenset().

    Returns
    -------
    np.ndarray[np.bool_]
        gate mask.

    """
    return np.isin(column, inset)


register_PyCode(isin_gate)

ISIN_gate = GateDef(isin_gate, tupledict(('inset', np.ndarray)), np.array([1,1]))


def percentile_gte_gate(col:np.ndarray[np.number], *, percentile:float):
    """
    Gate function for :attr:`PERCENTILE_GT_gate`

    Parameters
    ----------
    col : np.ndarray[np.number]
        array mask.
    percentile : float
        percecntile including and above which to have an element of col be True.

    Returns
    -------
    np.ndarray[np.bool_]
        gate mask.

    """
    return col >= np.percentile(col, percentile)


register_PyCode(percentile_gte_gate)


def percentile_gt_gate(col:np.ndarray[np.number], *, percentile:float):
    """
    Gate function for :attr:`PERCENTILE_GT_gate`

    Parameters
    ----------
    col : np.ndarray[np.number]
        array mask.
    percentile : float
        percecntile bove which an element of col will be True.

    Returns
    -------
    np.ndarray[np.bool_]
        gate mask.

    """
    return col > np.percentile(col, percentile)


register_PyCode(percentile_gt_gate)


def _normalize_percentile(param:dict[str,float])->dict:
    if 'percentile' not in param:
        param['percentile'] = 90.0
    if param['percentile'] <= 0.0 or param['percentile'] >= 100.0:
        raise ValueError("'percentile' must be in range of (0.0, 100.0)")
    return param


register_PyCode(_normalize_percentile)

PERCENTILE_GTE_gate = GateDef(percentile_gte_gate, tupledict(('percentile',float)), atomic=False, normalize=_normalize_percentile)
PERCENTILE_GT_gate = GateDef(percentile_gt_gate, tupledict(('percentile',float)), atomic=False, normalize=_normalize_percentile)

###############################################################################
################### Convenience functions for making gates  ###################
###############################################################################
def make_gt_gate(column:Column, mn:float, outside_expand:bool=False)->GateGroup:
    """
    

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mn : float
        minnimum value of gate (exclusive, ie greater than).
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
        gate = Gate(LIN_GT_gate, column, dict(vec=np.array([1.0]), m=float(mn)))
    else:
        gate =Gate(LIN_GT_gate, column, dict(vec=np.array([1.0]), m=float(mn)), 
                   expand=outside_expand)
    return GateGroup(_TT_ft, gate, title=f'{column.name()} > {mn}')

    
def make_geq_gate(column:Column, mn:float, outside_expand:bool=False)->GateGroup:
    """
    

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mn : float
        minnimum value of gate (inclusive, ie greater than or equal to).
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
        gate = Gate(LIN_GTE_gate, column, dict(vec=np.array([1.0]), m=float(mn)))
    else:
        gate =Gate(LIN_GTE_gate, column, dict(vec=np.array([1.0]), m=float(mn)), 
                   expand=outside_expand)
    return GateGroup(_TT_ft, gate, title=f'{column.name()} >= {mn}')


def make_lt_gate(column:Column, mx:float, outside_expand:bool=False)->GateGroup:
    """
    

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mx : float
        maximum value of column (exclusive, ie less than).
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
        gate = Gate(LIN_GTE_gate, column, dict(vec=np.array([1.0]), m=float(mx)))
    else:
        gate = Gate(LIN_GTE_gate, column, dict(vec=np.array([1.0]), m=float(mx)), 
                    expand= not outside_expand)
    return GateGroup(_TT_tf, gate, title=f'{column.name()} < {mx}')


def make_leq_gate(column:Column, mx:float, outside_expand:bool=False)->GateGroup:
    """
    Create a gate for column greater than mx.

    Parameters
    ----------
    column : Column
        Column on which gate is based.
    mx : float
        maximum value of column (inclusive, ie less than or equal to).
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
    return GateGroup(_TT_tf, gate, title=f'{column.name()} <= {mx}')


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
    return GateGroup.as_gategroup(Gate(ELLIPSOID_LTE_gate, (colx, coly), params))


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
    return ~Gate(ELLIPSOID_LTE_gate, (colx, coly), params)


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
    return GateGroup.as_gategroup(Gate(PERCENTILE_GTE_gate, column, dict(percentile=up)))


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
    return ~Gate(PERCENTILE_GTE_gate, (column, ), dict(percentile=low))