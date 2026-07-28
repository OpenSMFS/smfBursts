# -*- coding: utf-8 -*-
# Author : Paul David Harris
# email : harripd@gmail.com
# Created : 21/10/2025
"""
Load citations for smffretbursts

.. note::
    
    This script requires that all bibliographic fields have a DOI
"""
from importlib.resources import files
import re
import json

from .cite import register_citation, create_citation_group


_btdoiregex = re.compile(r'doi[^\S\r\n]*\=[^\S\r\n]*\{(([\w\d\-\./])+)\}')
_rissplrgx = re.compile(r'ER[^\S\r\n]*-[^\S\r\n]*')
_risdoiregex = re.compile(r'(DI|DO|DOI)[^\S\r\n]*-[^\S\r\n]*(([\w\d\-\./])+)\n')


def _doi_bib(record:str)->str:
    """Get doi from bibtex string"""
    return _btdoiregex.search(record).group(1)


def _split_ris(records:str)->list[str]:
    """split ris string of multiple records into list of individual record strings"""
    out = list()
    prev = 0
    for m in _rissplrgx.finditer(records):
        out.append(records[prev:m.span()[1]].strip())
        prev = m.span()[1]
    final = records[prev:]
    if final.strip():
        out.append(final.strip())
    return out


def _doi_ris(record:str)->str:
    """Get doi from ris string"""
    return _risdoiregex.search(record).group(2)

    
_bibtex = {_doi_bib(cite):f'@{cite}' for cite in 
           files('smfbursts').joinpath('citations/citations.bib').read_text(encoding='utf8').split('@') 
           if cite}
    
_risrefs = {_doi_ris(record):record for record in 
            _split_ris(files('smfbursts').joinpath('citations/citations.ris').read_text(encoding='utf8')) 
            if record}

_citations = json.loads(files('smfbursts').joinpath('citations/citations.json').read_text(encoding='utf8'))

_styles = {'bibtex':_bibtex, 'ris':_risrefs}


def _cite_kwargs(doi:str)->dict[str,str]:
    """Create kwargs of all styles speficied for given doi"""
    out = dict()
    if not doi.startswith('X'):
        out['doi'] = doi
    for style, records in _styles.items():
        if doi in records:
            out[style] = records[doi]
    return out

_fbccite = tuple(register_citation(tag, citation, **_cite_kwargs(doi)) 
                 for tag, (citation, doi) in _citations.items())

smfbursts_citations = tuple(f.citation for f in _fbccite)

create_citation_group('slidingwindowsearch', 'EggelingPNAS1988','FriesJPC1988', 'IngargiolaPLOSOne2016')
create_citation_group('flcs', 'GhoshMethods2018', 'FelekyanChemPhysChem2012')
create_citation_group('PIE', 'MullerBiophysJ2005', 'LaurencePNAS2005')