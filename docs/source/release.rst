Release Notes
-------------

0.3.0 (under development)
*************************
Additions

- :func:`smfbursts.fretfactory.make_bg` now adds ``"Start"``, ``"Start"`` and
  ``"Midtime"`` keys 
  (``"start"``, ``"stop"`` and ``"midtime"`` columns of :class:`sfmbursts.backgroundtables.Periods`)
- Add ``exclude_key``, ``exclude_key_pattern``, ``exclude_column`` keyword
  arguments to :func:`smfbursts.fretfactory.get_columns`
- Add :func:`smfbursts.cfuncs.kde_dist` function as faster version of :func:`smfbursts.cfuncs.kde_photons`
- Upgraded :class:`smfbursts.childphotontables.KDE` with the following:
    - Parameter added: ``"all_phot"`` for whether KDE is calculted just entire photon set
      or only on given time ranges.
    - Parameter added: ``"exclude_zero"`` for how to handle photon with 0 KDE in 
      *Tomov 2012* parameters
    - Column added: ``ph_kde`` giving photon-by-photon KDE values
    - Column added: ``tomovE``  column added to :class:`smfbursts.childphotontables.KDE`
    - ``tomovBR``  column added to :class:`smfbursts.childphotontables.KDE`
- Add the following dictionary like methods to :class:`smfbursts.datamodel.utils.SequenceDefaults`:
    - :meth:`smfbursts.datamodel.utils.SequenceDefaults.keys`
    - :meth:`smfbursts.datamodel.utils.SequenceDefaults.values`
    - :meth:`smfbursts.datamodel.utils.SequenceDefaults.items`
- :attr:`smfbursts.ph_sel.ChannelType` type hint added, a literal for the different
  photon channel categories
- :attr:`smfbursts.photondata.PhotonData.stream_names` property added, allows
  setting default names for streams and channels that override the infered channel
  names derived from setup when calling :meth:`smfbursts.photondata.PhotonData.get_stream_names`


Changes

- ``smfbursts.childphotontables.laplace_kde_2cde`` renamed :func:`smfbursts.childphotontables.laplace_kde`
- ``smfbursts.childphotontables.gaussian_kde_2cde`` renamed :func:`smfbursts.childphotontables.gaussain_kde`
- ``smfbursts.childphotontables.register_2cde_func`` renamed :func:`smfbursts.childphotontables.register_kde_func`


Fixes

- Fix :func:`smfbursts.cfunc.kde_photons` bug where function call 
  ``sfmbursts.cfuncs.kde_photons(ta, tau, tb drop_self=True)`` gave same
  result as ``sfmbursts.cfuncs.kde_photons(ta, tau, tb drop_self=False)``
- Fix ``alex2cde`` column computation
- Fix typevalidator for numpy object array passing 0 size arrays with impropper type checking
- Fixed memory leaks in ``smfbursts.cfuncs`` stemming from not decreffing after using ``PyTuple_Pack``


Internal Changes/Additions and bug fixes

- Added :class:`smfbursts.datamodel.utils.CheckDict` and :class:`smfbursts.datamodel.utils.CheckNestDict`
  classes, which are dict subclasses which enforece key and value types, the nest
  for also forces keys to be tuples, which can read keys as nested within each other.
- :class:`smfbursts.childphotontables.KDE` now uses
- Refactor ``smfbursts.cfuncs`` (note this was also part of fixing the memory leaks):
    - Replace ``PyArrayList`` with ``PyArrayInt64List`` and ``PyArrayDoubleList``
    - added ``get_dset`` fucntion to cover infering detector accross multiple functions
    - rename ``converter_listofarrays`` to ``converter_listofarrays_int64``
      and add functionality to treat single 1D array as 1-length list of 1D arrays
    - added ``init_arraylist_int64/double`` functions for ``0/NULL``-ing ``PyArrayInt64List/DoubleList`` structures


0.2.1
*****

Additions

- ``rcParams`` has ``"analysis.lifetime`` key to speficy if fretfactory should
  assume data is pulsed
- :func:`smfbursts.fretfactory.make_fret_from_base` now can add ``"Nmbg_DD"`` 
  key (``"nanomean_bg"`` column)

Changes

- Fix issues with plotting function not correctly combining kwargs.
  This was a fix to an internal utils function.


0.2.0
*****

Additions

- ``rotational_corr_c`` column added to 
  :class:`Ratios <smfbursts.childphotontables.Ratios>`
- ``l1`` and ``l2`` optional parameters added to 
  :class:`Ratios <smfbursts.childphotontables.Ratios>`
  to support correction for mixing between polarizations in high NA setups
- :class:`Phasor <smfbursts.lifetimetables.Phasor>` added for phasor analysis of bursts
- add functions for fitting fluoresence decays


Changes

- ``nanomean`` and ``nanomean_bg`` columns now share same 2 keys: 
  ``(phsel:PhSel, irstyle:{"thresh", "mean", "max"})``
  The key ``irfstyle`` has the following options ``"mean"``, ``"max``" and ``"thresh"`` .
  The first two specify IRF by the value in the 
  :attr:`PhotonData.irf <smfbursts.photondata.PhotonData.irf>`, the last by
  :attr:`PhotonData.irf_thresh <smfbursts.photondata.PhotonData.irf_thresh>`


Internal Changes

- Improved and unified handling of ``irfstyle`` for ``nanomean_bg``, ``nanomean``, and
  :class:`Phasor <smfbursts.lifetimetables.Phasor>` parameters.


0.1.3
*****

Internal alterations only

- fix docstrings in cfuncs
- Rewrite some utils functions to ensure concsistency across platforms


0.1.2
*****

First full release.
Tests have run sucessfully on Linux, Mac and Windows platforms.
