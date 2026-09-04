Introduction
============

Installation
------------

Installation from PyPi is simple:

.. code-block:: bash

    pip install smfbursts


The repository is still under review for CondaForge, so the following hopefully will work in the future, but not now

.. code-block:: bash

    conda install smfbursts -c conda-forge

You can also install from source:

.. code-block:: bash
   pip install git+https://github.com/OpenSMFS/smfBursts.git



Usage
-----

smfBursts works on 4 core objects: :class:`PhotonData <smfbursts.photondata.PhotonData>`, 
:class:`Param <smfbursts.datamodel.tables.Param>`, 
:class:`Column <smfbursts.datamodel.tables.Column>` and :class:`GateGroup <smfbursts.datamodel.tables.GateGroup>`.

The base measured data is stored in :class:`PhotonData <smfbursts.photondata.PhotonData>`.
Specific analysis techniques are specified with :class:`Param <smfbursts.datamodel.tables.Param>`
Specific parameters from analysis are accessed with :class:`Column <smfbursts.datamodel.tables.Column>`.
Specific subpopulations are selected from :class:`Param <smfbursts.datamodel.tables.Param>` 
and :class:`Column <smfbursts.datamodel.tables.Column>` with 
:class:`GateGroup <smfbursts.datamodel.tables.GateGroup>`.

For instance, a burst search can be specified in a 
:class:`Param <smfbursts.datamodel.tables.Param>` as follows:


>>> bursts = smf.Param(bg=bg, m=10, F=6.0, streams=smf.PhSel('0ex_1ex1em'))


``bg`` is a :class:`Param <smfbursts.datamodel.tables.Param>` defining a computation of background,
while the remaining burst search sliding window size and snr threshold are specified in the ``m`` and ``F`` arguments.

Then specific data like photon counts can be specified with 
:class:`Column <smfbursts.datamodel.tables.Column>` in a similar way

>>> n_dd = smf.Column(bursts, 'nph_raw', smf.PhSel('0ex0em'))


The data can be accessed from a :class:`PhotonData <smfbursts.photondata.PhotonData>` object ``data`` 
with the :meth:`DataSet.get_column() <smfbursts.datamodel.tables.DataSet.get_column>` method

>>> data.get_column(n_dd)
array([50, 21, 54, ..., 78, 24])


And gates can be used to filter this data

>>> gateD50 = smf.make_geq_gate(n_dd, 50.0)
>>> data.get_column(n_dd, gate=gateD50)
array([50, 54, 53, ..., 61m 78])



Finally, citations are accessible with the 
:func:`get_citations() <smfbursts.cite.citations.get_citations>` function.
