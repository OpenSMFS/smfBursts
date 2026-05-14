# smfBursts

## Project Description

smfBursts is an open source package for burst analysis of freely-diffusing single molecule experiments.
It seeks to be highly reproducible, general and easy to use.
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
- Flexible and reporoducible burst selection based on logical operations on arbitrary numbers of gates
- Large suite of burst statistics, such as E, S, peak rate, brightness, mean fluorescence lifetime, etc.


## Install

Currently the best way to install is to clone the git repo onto your computer

```bash
pip install .
```

## License and Copyrights

License: GNU GPL 2
You cna find a full copy fo the license in the file LICENSE.txt
