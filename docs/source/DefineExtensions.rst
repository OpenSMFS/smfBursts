.. _defext:

smfBursts extensions
====================

.. module:: smfbursts

New |Table| objects are defined by subclassing either |BaseTable| or |ChildTable|.
These will create their respective type of |Table|
and the ability to create the associated |Param| objects defining such |Table| objects.

Basic Table Definition
----------------------

Below is the basic structure of a new |ChildTable|.
This outlines the 3 basic ways of computing columns, as well as how parents, params and columns are defined.


.. code-block::

    from collections.abc import Iterator

    import numpy as np

    from smfbursts.datamodel.tables import BaseTable, ChildTable, ParamDef, ParentDef, ColumnDef
    from smfbursts.datamodel.immutabledata import TV_int

    class Foo(BaseTable):
        # define parents (BaseTables do not need any)
        parent_defs = tuple()

        # define params needed
        param_defs = (ParamDef('bins', TV_int(mn=1)), )

        # define columns
        column_defs = (ColumnDef('bns', tuple(), store='all', dtype=np.int64), )

        def __init_columns__(self):
            self._add_column('bns', np.arange(self.param.params['bins']))

    class Bar(ChildTable):
        # defines the parent parameters, and what type of table the parent must be
        parent_defs = (ParentDef('base', Foo, is_base=True), )

        # defines the needed params, and their types, and limits on their values
        param_defs =  (ParamDef('a', TV_int(mn=0)), )

        # defines the columns, type, offset, how stored, and how to retrieve
        column_defs = (ColumnDef('bor', tuple(), store='all'),
                       ColumnDef('ber', (TV_int(mn=0), ), store='some', get_func='_get_bir'),
                       ColumnDef('bir', (TV_int(mn=0), ), iter_func='_get_ber'))

        def __init_columns__(self):
            # initialize column defined upon table creation
            col = self.parents['base']['istop'] % self.param.params['a'] # dummy calculation
            self._add_column('ini', tuple(), col)

        def _get_ber(self, key:int)->np.ndarray[np.float64]:
            # get a column when column is requested
            # get start times
            istart = self.parents['base']['istart']
            # get parameter value
            a = self.param.params['a']
            out (istart + a) % key # dummy calculations
            if key == 1: # example of saving some-times
                self._add_column('ber', (key,), out)
            return out

        def _iter_bir(self, key:int)->Iterator[float]:
            # iterate over each row in a column, useful to save memory
            a = self.param.params['a']
            for c in self.parents['base'].iter_column('istart'):
                yield c - a


In the above example, 3 different ways of computing a column are shown.

#. Column 'ini': You can define a column in the 
   :meth:`Table.__init_columns__ <smfbursts.datamodel.Table.__init_columns__>`
   method, where the column must be set using the 
   :meth:`Table._add_column <smfbursts.datamodel.tables.Table._add_column>` method.
#. The column can be created when they are requested.
   If the |ColumnDef| has ``get_func`` specified, 
   that method will be called to get the entire column.
#. The column values can be computed by an iterator.
   The |ColumnDef| must have an ``iter_func`` specified, 
   that method will be used to iterate over the column.


Some notes:

#. The |Table| object already has the logic to store iterations if the given column needs to be stored,
   or returned not as in iterator but as an array.
#. The convention throughout all official smfBursts code is that
   methods used to get a column are named ``_get_<column-name>``,
   while those used to iterate over a column's values are names ``_iter_<column-name>``.
#. When defining a |BaseTable| it is best practice to have at least one column set in the
   :meth:`Table.__init_columns__ <smfbursts.datamodel.Table.__init_columns__>`
   method, so that upon table creation, the size of the table is known.
#. When the current table depends on the column of a parent (the purpose of parent tables in the first place),
   it is highly recomended to access column using the ``self.parents['<parent-name>'][col-name, keys...]``
   syntax. The :attr:`Table.parents <smfburts.datamodel.tables.Table.parents>` attribute allows access to the |Table|
   objects of the parents defined in the |Param|, so getting the correct column array is very easy.


|BasePhotonTable| and |ChildPhotonTable| subclasses
---------------------------------------------------

Currently the only type of |DataSet| (which is intented as an abstract base class), is |PhotonData|.
For this type of data, often the rows of your tables are defined by ranges of time, (like bursts).
So you will usually define |BaseTable| subclasses using |BasePhotonTable| subclasses.
Likewise, |ChildTable| subclasses using |ChildPhotonTable|

Below is an example piece of code,
giving 2 examples of how to define |BasePhotonTable| subclasses,
and 1 example fo how to define |ChildPhotonTable| subclasses

.. code-block::

    from collections.abc import Iterator

    import numpy as np

    from smfbursts.datamodel.tables import (
        ParamDef, ParentDef, ColumnDef, Param, paramproperty
        )
    from smfbursts.datamodel.immutabledata import TV_float

    ################################## NEW IMPORTS ################################
    from smfbursts.photondata import (
        BasePhotonTable, ChildPhotonTable, make_base_column_defs
        )
    from smfbursts.ph_sel import DetDef, TV_DetDef, PhSel
    from smfbursts.background import BG
    import smfbursts.cfuncs as smc
    ###############################################################################

    # Define a base table that has no parents, must set detdef ####################
    class SpacesRoot(BasePhotonTable):
        """Example BasePhotonTable that has no parents, defines ranges of time
        of time timeon, separated by times of timeoff"""
        parent_defs = tuple()
        param_defs = (
            # Need detdef for table that has no parents ###########################
            ParamDef('detdef', TV_DetDef),
            ParamDef('timeon', TV_float(mn=0.0)),
            ParamDef('timeoff', TV_float(mn=0.0)),
                  )

        # Automatically generate
        column_defs = make_base_column_defs()

        def __init_columns__(self):
            # compute index partitions, first convert s to clk_p units
            timeon = int(self.param.timeon * self.origin.clk_p) 
            timeoff = int(self.param.timeoff * self.origin.clk_p)
            ttot = timeon + timeoff # compute time of enteire "period"
            # get min/max times
            tmin = self.origin.times[0]
            tmax = self.origin.times[-1]
            # create arrays of start and stop times
            start = np.arange(tmin, tmax, ttot)
            stop = np.arange(tmin+timeon, tmax, ttot)
            start = start[:stop.size] # ensure start and stop are the same size
            # Set start and stop columns ##########################################
            self._add_column('start', tuple(), start)
            self._add_column('stop', tuple(), stop)
            # Compute istart/istop this should be a general pattern for ###########
            # BasePhotonTable #####################################################
            istart, istop = smc.index_ranges(self.origin.times, start, stop)
            # Add istart/istop columns, needed for complete BasePhotonTable #######
            self._add_column('istart', tuple(), istart)
            self._add_column('istop', tuple(), istop)

    # Define a base table with parents, must have detdef paramproperty ############
    class SpacesBranch(BasePhotonTable):
        """Example BasePhotonTable that has parents, defines ranges of time
        of time timeon, separated by times of timeoff"""
        param_defs = (
            ParamDef('timeon', TV_float(mn=0.0)),
            ParamDef('timeoff', TV_float(mn=0.0)),
                  )
        parent_defs = (ParentDef('bg', BG), )

        def __init_columns__(self): # this is the same as SpacesRoot
            # compute index partitions, first convert s to clk_p units
            timeon = int(self.param.timeon * self.origin.clk_p) 
            timeoff = int(self.param.timeoff * self.origin.clk_p)
            ttot = timeon + timeoff # compute time of enteire "period"
            # get min/max times
            tmin = self.origin.times[0]
            tmax = self.origin.times[-1]
            # create arrays of start and stop times
            start = np.arange(tmin, tmax, ttot)
            stop = np.arange(tmin+timeon, tmax, ttot)
            start = start[:stop.size] # ensure start and stop are the same size
            # Set start and stop columns ##########################################
            self._add_column('start', tuple(), start)
            self._add_column('stop', tuple(), stop)
            # Compute istart/istop this should be a general pattern for ###########
            # BasePhotonTable #####################################################
            istart, istop = smc.index_ranges(self.origin.times, start, stop)
            # Add istart/istop columns, needed for complete BasePhotonTable #######
            self._add_column('istart', tuple(), istart)
            self._add_column('istop', tuple(), istop)
        # BasePhotonTable with parent(s), must define the detdef paramproperty
        @paramproperty
        def detdef(self, param:Param)->DetDef:
            return param.parents['bg'].detdef

    # define child photon tabletable ##############################################
    class BarTab(ChildPhotonTable):
        """Simple example ChildPhotonTable, has 1 column, clump,
        which has the value of the detector with the most counts per burst"""
        parent_defs = (ParentDef('base', BasePhotonTable, is_base=True), )
        param_defs = tuple()
        column_defs = (ColumnDef('clump', (PhSel, ), 0, dtype=np.int64,
                                 iter_func='_iter_clump'), )

        def _iter_clump(self, phsel:PhSel)->Iterator[int]:
            for phdets in self.parents['base'].iter_column('ph_dets', phsel):
                yield np.max(np.bincount(phdets))


You will see the |ChildPhotonTable| class requires no more or less definition than before.
It should be noted that the ``parent_defs`` |ParentDef| marked as``is_base=True`` 
needs to be a |BasePhotonTable| or |ChildPhotonTable|.

#. Have a parameter named ``detdef`` which is a |DetDef| (less common)
#. Have a |paramproperty| named ``detdef`` which gets the |DetDef| 
   (more common, usually from a specific parent)

What is a |paramproperty|? That lead us right to the next section.


Special |Table| method decorators
---------------------------------

There are 4 decorators that are specifically designed to be used on |Table| methods.
This give extended property_ and classmethod_ like behavior to these methods.
These are

#. |paramproperty| signature: ``def param_property(cls, param:Param):``
#. |parammethod| signature: ``(cls, param:Param, *args, **kwargs)``
#. |tableproperty| signature: ``(cls, param:Param, origin:DataS, *args, **kwargs)``
#. |tablemethod| signature: ``(cls, param:Param, origin:DataS, *args, **kwargs)``


These all implement behavior that can be used by either the table,
or |Param| objects based on the given |Table| type.
In the case of |paramproperty| these methods should only depend on the values
defining the |Param|, ie information about the data creating the table is irrelevant.
These methods behave like of **either** |Table| or |Param| objects.

So, if we have a |Param| based on the previously described ``BarTab`` table we can
do the following

>>> btab_param.detdef
DetDef2ex2em at 0x74cccc0463e0


or, if we have a table of type ``BarTab``

>>> btab_table.detdef
DetDef2ex2em at 0x74cccc0463e0


|tableproperty| methods are assumed to depend both on the data and |Param| values.
Therefore, they can be accessed like properties of their |Table|,
but become methods requiring 1 argument, of a |PhotonData| object of |Param| objects.

|parammethod| and |tablemethod| are methods that need more information than
the |Param| or |PhotonData|, and so are always methods for either their
|Table| or |Param| instances.
The difference between them is that |parammethod| should not require a |PhotonData|
object to compute, while |tablemethod| needs everything.

See their documentation to see examples of how each behaves
with full function signatures.

.. |DataSet| replace:: :class:`DataSet <smfbursts.datamodel.tables.DataSet>`
.. |Table| replace:: :class:`Table <smfbursts.datamodel.tables.Table>`
.. |BaseTable| replace:: :class:`BaseTable <smfbursts.datamodel.tables.BaseTable>`
.. |ChildTable| replace:: :class:`ChildTable <smfbursts.datamodel.tables.ChildTable>`
.. |ParentDef| replace:: :class:`ParentDef <smfbursts.datamodel.tables.ParentDef>`
.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |ParamDef| replace:: :class:`ParamDef <smfbursts.datamodel.tables.ParamDef>`
.. |Column| replace:: :class:`Column <smfbursts.datamodel.tables.Column>`
.. |ColumnDef| replace:: :class:`ColumnDef <smfbursts.datamodel.tables.ColumnDef>`
.. |Coparam| replace:: :attr:`Column.origin_param <smfbursts.datamodel.tables.Column.origin_param>`
.. |Cbparam| replace:: :attr:`Column.base_param <smfbursts.datamodel.tables.Column.base_param>`
.. |paramproperty| replace:: :class:`@paramproperty <smfbursts.datamodel.tables.paramproperty>`
.. |parammethod| replace:: :class:`@paramproperty <smfbursts.datamodel.tables.parammethod>`
.. |tableproperty| replace:: :class:`@paramproperty <smfbursts.datamodel.tables.tableproperty>`
.. |tablemethod| replace:: :class:`@paramproperty <smfbursts.datamodel.tables.tablemethod>`
.. |PhotonData| replace:: :class:`PhotonData <smfbursts.photondata.PhotonData>`
.. |BasePhotonTable| replace:: :class:`BasePhotonTable <smfbursts.photondata.BasePhotonTable>`
.. |ChildPhotonTable| replace:: :class:`BasePhotonTable <smfbursts.photondata.ChildPhotonTable>`
.. |PhSel| replace:: :class:`PhSel <smfbursts.ph_sel.PhSel>`
.. |DetDef| replace:: :class:`DetDef <smfbursts.ph_sel.DetDef>`
.. |Periods| replace:: :class:`Periods <smfbursts.background.Periods>`
.. _property: `property <https://docs.python.org/3/library/functions.html#property>`__
.. _classmethod: `classmethod <https://docs.python.org/3/library/functions.html#classmethod>`__
