Release Notes
-------------

0.2.1
-----

Additions

- ``rcParams`` has ``"analysis.lifetime`` key to speficy if fretfactory should
  assume data is pulsed
- :func:`smfbursts.fretfactory.make_fret_from_base` now can add ``"Nmbg_DD"`` 
  key (``"nanomean_bg"`` column)

Changes

- Fix issues with plotting function not correctly combining kwargs.
  This was a fix to an internal utils function.

0.2.0
-----

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
-----

Internal alterations only

- fix docstrings in cfuncs
- Rewrite some utils functions to ensure concsistency across platforms


0.1.2
-----

First full release.
Tests have run sucessfully on Linux, Mac and Windows platforms.
