smfBursts
==========

smfBursts is an open source library for analyzing freely diffusing confocal single molecule fluorescence spectroscopy data.
It seeks to implement the state of the art algorithms in a transparent, simiple and reproducible manner.

Core Benefits
-------------

Broad Support
*************

smfBursts is able to handle most excitaiton and detection schemes,
from single excitation, ALEX_, PIE_ (aka nsALEX_), MFD_ and `3-color`_ schemes.

Further more, various methods, from stanard E/S ratio based analysis, mean lifetime, and BVA_ are
fully supported without need of writing custom functions.

Citations
*********

Importantly smfBursts provides simple ways of obtaining a full description of all parameters and steps used in data analysis.
This allows for supplementary material to describe without code the exact data analysis parameters used.

Additionally the :func:`smfbursts.cite.citations.get_citations` function provides an easy way to obtain
citations for each method used in the current environment.


Current Install
---------------

Install by downloading the git repo, navigating into the top level directory of said repo and running the following command.

``pip install .``

Contents
========

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    Introduction
    Tutorials
    UserGuides
    Documentation
    release
    contributing

.. _hellenkamp: https://doi.org/10.1038/s41592-018-0085-0
.. _ALEX: https://doi.org/10.1073/pnas.0401690101
.. _PIE: https://doi.org/10.1529/biophysj.105.064766
.. _nsALEX: https://doi.org/10.1073/pnas.0508584102
.. _MFD: https://doi.org/10.1016/S0168-1656(00)00412-0
.. _3-color: https://doi.org/10.1021/acs.jpcb.8b07768
.. _BVA: https://doi.org/10.1016/j.bpj.2011.01.066
