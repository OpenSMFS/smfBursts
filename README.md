# smfBursts

[![Tests](https://github.com/OpenSMFS/smfBursts/actions/workflows/test.yml/badge.svg)](https://github.com/OpenSMFS/smfBursts/actions)
[![Documentations Status](https://readthedocs.org/projects/smfbursts/badge?version=latest)](https://smfbursts.readthedocs.io/en/latest/?badge=latest)

## Project Description

smfBursts is an open source package for analysis of freely-diffusing single molecule experiments, with a focus on burst-wise analysis techniques.

The goal is to be easy to use, easy to reproduce, and easy to communicate

While initially designed for FRET based experiments,
it supports any number of spectral, as well as polarization based detection channels,
as well as any number of excitation channels.

### smfBursts and Reproducibility

smfBursts is an effort to bring 
[reproducible computing](http://dx.doi.org/10.1371/journal.pcbi.1003285)
to the field of single-molecule confocal microscopy.
smfBursts records all instructions with their thresholds and other parameters
in special `Param` objects, which have a standard text representation,
so that exact computations can be unambiguously reproduced.

smfBursts provides tutorials for use with
[Jupyter Notebooks](https://jupyter.org),
and plotting relies on the ubiquitous python plotting library
[matplotlib](https://matplotlib.org/)

## Supported Features

smfBursts provides algorithms for the following analysis methods
- Backgrounds estimation as a function of time
- Sliding-window bursts earch with adaptive (background-dependent) rate-threshold
- Multi-channel burst search with logical combinations of bursts, extending DCBS
- Burst corrections based on [Hellenkamp et. al.](https://doi.org/10.1038/s41592-018-0085-0), and the flexibility to specify corrections for other excitation/detector schemes with more channels and polarization (MFD)
- Flexible and reproducible burst selection based on logical operations on arbitrary numbers of gates
- Large suite of burst statistics, such as E, S, peak rate, brightness, mean fluorescence lifetime, etc.


## Install

smfBursts is available on pypi, install with

```bash
pip install smfbursts
```

hopefully soon smfbursts will also be available on conda-forge

```bash
conda install smfbursts -c conda-forge
```

## License and Copyrights

License: MIT

You can find a full copy fo the license in the file LICENSE.txt

### A request from the developer

Be nice, give credit and give back.

That is the basic philosophy I would like users of this package to adopt.

If you are a lab group, a company selling equipment or whoever/whatever else,
if you can make use of smfBursts, I want you to use it,
but please don't box people out, and give me the proper citation.
I chose the MIT license because I didn't want to scare anyone (especially companies) away because of strong copyleft licenses like GPL.

MIT basically lets anyone do pretty much whatever they want with this code.
I would however ask people to use this in a more copyleft way, making sure to give credit, and make open source what they develop off of it.

Open source is incredibly valuable to the scientific community.
It makes analysis transparent and reproducible.
There's nothing more frustrating than reading in a paper
that the analysis was done with some black box piece of software.
Open source lets use your new methods,
and let's us understand what you are doing, and check your work.

So if you are doing science, or selling to those doing science, I ask you keep that in mind.

If you don't act that way, the MIT license can't stop you,
but you will get me wagging a finger at you in conversation.