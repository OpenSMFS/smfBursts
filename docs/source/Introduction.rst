Introduction
============

Installation
------------

In the future, FRETBursts will be available through PIP and conda-forge,

currently you can install FRETBursts locally like so:


.. code-block:: bash

    git clone FretTables
    cd FretTables
    pip install -e .


Eventually install

.. code-block:: bash

    pip install fretbursts

.. code-block:: bash

    conda install fretbursts -c conda-forge


Usage
-----

.. module:: fretbursts

FRETBursts works on 5 core objects: :class:`PhotonData <photondata.PhotonData>`, 
:class:`Param <datamodel.tables.Param>`, 
:class:`Column <datamodel.tables.Column>` and :class:`GateGroup <datamodel.tables.GateGroup>`.

The base measured data is stored in :class:`PhotonData <photondata.PhotonData>`.
Specific analysis techniques are specified with :class:`Param <datamodel.tables.Param>`
Specific parameters from analysis are accessed with :class:`Column <datamodel.tables.Column>`.
Specific subpopulations are selected from :class:`Param <datamodel.tables.Param>` 
and :class:`Column <datamodel.tables.Column>` with 
:class:`GateGroup <datamodel.tables.GateGroup>`.

For instance, a burst search can be specified in a 
:class:`Param <datamodel.tables.Param>` as follows:


>>> bursts = frb.Param(bg=bg, m=10, F=6.0, streams=frb.PhSel('0ex_1ex1em'))


``bg`` is a :class:`Param <datamodel.tables.Param>` defining a computation of background,
while the remaining burst search sliding window size and snr threshold are specified in the ``m`` and `F`` arguments.

Then specific data like photon counts can be specified with 
:class:`Column <datamodel.tables.Column>` in a similar way

>>> n_dd = frb.Column(bursts, 'nph_raw', frb.PhSel('0ex0em'))


The data can be accessed from a :class:`PhotonData <photondata.PhotonData>` object ``data`` 
with the :meth:`datamodel.DataSet.get_column() <datamodel.tables.DataSet.get_column>` method

>>> data.get_column(n_dd)
array([50, 21, 54, ..., 78, 24])


And gates can be used to filter this data

>>> gateD50 = frb.make_geq_gate(n_dd, 50.0)
>>> data.get_column(n_dd, gate=gateD50)
array([50, 54, 53, ..., 61m 78])



Finally, citations are accessible with the 
:func:`get_citations() <cite.citations.get_citations>` function.
