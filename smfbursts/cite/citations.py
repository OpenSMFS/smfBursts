# -*- coding: utf-8 -*-
# Author : Paul David Harris
# email : harripd@gmail.com
# created : 21/10/2025
# purpose: managing citations
r"""
Citations
=========

Module provides a way of tracking which functions, have been called, and therefore
which citations should be included in any papers.

For most users, it is as simple as::
    
    import smfbursts as smf
    ['do all your analysis here', ...]
    citations = smf.get_citations()
    for citation in citations:
        print(citation)

and the output will be the citations that should be used in a publication using
the code.

.. note::
    
    This should be seen as the minimal set of citations, ie citations in this
    list are those that reflect the algorithms/methods used. Authors are still
    responsible for including appropriate citations based on additional factors.


If writing your own extension to smfBursts, there are 3 more functions that are
important\:

#. :func:`register_citation` add a new citation to the citation database
#. :func:`create_citation_group` specify new group of citation (citation-group), defining a set of citations that often go together.
#. :func:`cite` a decorator for adding citation(s) to the list of citations retrieved by :func:`get_citations` when the decorated function is called.
#. :func:`add_citation` when called, adds citation(s) to the list of citations retrieved by :func:`get_citations`.


basic usage::
    
    import smfbursts as smf
    
    smf.register_citation("CleeseNature1970", "Cleese, J. Fluorescent properties of Spam. Nature (1970) 123. 21", 
                          doi="10.1000/nature.123.21", bibtex=".", ris=".")
    
    smf.register_citation("IdleScience1969", "Idle, E. Mortality of Domesticated Mopsitta Tanta. Science. (1969) 322. 55".
                          doi="10.1111/science.322.55", bibtex="..", ris="..")
    
    smf.register_citation("PalinCell1970", "Palin, M. Supression of Diffusivity in SPA15 by Inq. PNAS, (1970) 999, 11",
                          doi="10.1222/pnas.999.11", bibtex="...", ris="...")
    
    smf.create_citation_group("monty", "IdleScience1969", "PalinCell1970")
    
    @smf.cite("IdleScience2027", purpose="dead")
    def norway(*args):
        return 'blue'
    
    def cheese(args):
        if len(args) > 2:
            smf.add_citation("monty", "IdleSciecne2027", purpose="meaning of life")
        return "cheddar"
    for citation in smf.get_citations(include_purpose):
        print(citation)

| ("Cleese, J. Fluorescent properties of Spam. Nature (1970) 123. 21", ("dead", "meaning of life"))
| ("Idle, E. Mortality of Domesticated Mopsitta Tanta. Science. (1969) 322. 55", ("meaning of life",))
| ("PalinCell1970", "Palin, M. Supression of Diffusivity in SPA15 by Inq. PNAS, (1970) 999, 11", ("meaning of life", ))
    
"""
from functools import wraps
from collections.abc import Callable, Sequence
from typing import Union, ClassVar
import warnings
from itertools import chain, repeat
from textwrap import wrap


class Citation:
    """
    Object representing single reference/citation. Contains a short name tag str
    for referencing with :func:`cite` and :func:`add_citation`, the full citation
    string, and dictionary of additional styles (default bibtex and ris)
    
    Parameters
    ----------
    tag : str
        short name for citation, the standard format is AuthorJournalYear, 
        where Author is the last name of the first author, Journal is the standard
        abreviation (without periods) for the journal.
    citation : str
        Full string citation. The standard is the Nature-format
    sytles : dict[str:str]
        Dictionary of *citation-style* keys and *citation* values.
        The standard is to suply "bibtex" and "ris"styles.
    
    """
    _prefered:ClassVar[str] = None
    _tag:str
    _citation:str
    _styles:dict[str,str]
    
    def __init__(self, tag:str=None, citation:str=None, 
                 styles:dict[str,str]=None):
        if not (tag or citation or styles):
            raise ValueError("empty citation")
        self._tag = tag if tag is None else str(tag)
        self._citation = citation if citation is None else str(citation)
        styles = dict() if styles is None else dict(styles)
        if not isinstance(styles, dict):
            raise ValueError("styles must be specified as a dictionary")
        styles = {str(key):str(val) for key, val in styles.items()}
        if Citation._prefered is None and len(styles) == 1:
            Citation._prefered = list(styles.keys())[0]
        self._styles = styles
        if not self.cited:
            warnings.warn(f"citation with tag: {tag} has no citation information")
        
    def __getitem__(self, key):
        if key.startswith('_'):
            raise KeyError("cannot access private keys")
        return getattr(self, key)
    
    @property
    def complete(self)->bool:
        """If the citation has both a tag and some form of citation"""
        return self._tag is not None and self.cited
    
    @property
    def cited(self)->bool:
        """If citation contains some form of citations"""
        return self._citation or self._styles
    
    @property
    def tag(self)->str:
        """Tag str of citation"""
        if self._tag is None:
            raise AttributeError("tag not set")
        return self._tag
    
    @tag.setter
    def tag(self, val):
        if val is None:
            return
        val = str(val)
        if self._tag is not None and self._tag != val:
            raise AttributeError("tag already set")
        if self._tag is None:
            self._tag = val
    
    @property
    def citation(self)->str:
        """String citation, Nature format"""
        if self._citation is None:
            raise AttributeError("citation not set yet")
        return self._citation
    
    @citation.setter
    def citation(self, val:str):
        if val is None:
            return
        val = str(val)
        if self._citation is not None and self._citation != val:
            raise AttributeError("citation already set, cannot ")
        if self._citation is None:
            self._citation = val
    
    def replace_citation(self, citation:str)->None:
        """
        Change the citation string.

        Parameters
        ----------
        citation : str
            New string for citation.

        """
        citation = str(citation)
        self._citation = citation
    
    @classmethod
    def set_prefered_style(cls, name:str)->None:
        """
        Set the type of reference to return when 
        accessing :attr:`Citation.prefered_style`.

        Parameters
        ----------
        name : str
            name of reference type to set as :class:`Citation.prefered_style`.

        """
        cls._prefered = str(name)
        
    def get_style(self, style:str)->str|None:
        """
        Retrive the given style of citation. If given style not set, return None.

        Parameters
        ----------
        style : str
            Style of citation to retrieve, standard styles supplied are "bibtex"
            and "ris".

        Returns
        -------
        None | str
            Citation string of given style.

        """
        return self._styles.get(style, self.prefered_style)
    
    @property
    def prefered_style(self)->None|str:
        """The best citation type, based on prefered style"""
        if Citation._prefered in self._styles:
            return self._styles[Citation._prefered]
        if self._citation is not None:
            return self._citation
        if self._styles:
            return list(self._styles.keys())[0]
        return self._tag
        
    def set_style(self, name:str, citation:str, overwrite:bool=False, warn:bool=True)->None:
        """
        Set a ``citation`` string for style ``name``.

        Parameters
        ----------
        name : str
            Citation style.
        citation : str
            Citation string.
        overwrite : bool, optional
            If True, allow overwriting existing string, otherwise raise error.
            The default is False.
        warn : bool, optional
            Whether to produce a warning when overwriting citation style.
            The default is True.

        Raises
        ------
        ValueError
            style ``name`` already set.

        """
        name, citation = str(name), str(citation)
        if name in self._styles and self._styles[name] != citation:
            if not overwrite:
                raise ValueError(f"style {name} already set")
            if warn:
                warnings.warn(f"changing citation of style {name}")
        self._styles[name] = citation
    
    def update_style(self, overwrite:bool=False, warn:bool=True, **kwargs:str)->None:
        """
        Add multiple styles at once.

        Parameters
        ----------
        overwrite : bool, optional
            If True, allow overwriting of existing styles, otherwise raise error.
            The default is False.
        warn : bool, optional
            Whether to warn if overwriting existing style. The default is True.
        **kwargs : str
            Citation string according to specified style.

        """
        for name, citation in kwargs.items():
            self.set_style(name, citation)
    
    def as_dict(self)->dict[str,str]:
        """Create dictionary of all citation types, including tag, into dictionary"""
        out = dict()
        if self._tag is not None:
            out['tag'] = self._tag
        if self._citation is not None:
            out['citation'] = self._citation
        if self._styles is not None:
            out.update(self._styles)
        return out
    
    def __str__(self):
        out = 'Citation:'
        if self._tag:
            out +=  f'tag: {self._tag}'
        if self._citation:
            out += f', citation: {self._citation}'
        if self._styles:
            out += "\nwith the following formats: "
            for name in self._styles.keys():
                out += f'{name}, '
            out = out[:-2]
        return out


class CitedReason:
    """
    Class for containing a :class:`Citation`, in :attr:`CiteReason.cites` and
    a set of "purposes" (string description of reason for citations), in the
    :attr:`Citation.purpose`. All fields of :class:`Citaion` are mirrored so that
    it can be treated like an extension of :class:`Citation`.
    
    Parameters
    ----------
    cites : Citation
        The citation being wrapped.
    purpose : str | Sequence[str]
        The purpose(s) for which the citation is being used. If a string, converted
        into a set with one element (that of the string). Purpose is always
        converted to string. The default is None.
    
    """
    __slots__ = ('_cites', '_purpose')
    _cites:Citation
    _purpose:set[str]

    def __init__(self, cites:Citation, purpose:Union[str, Sequence[str]]=None):
        self._cites = cites
        purpose = set() if purpose is None else purpose
        purpose = {purpose,} if isinstance(purpose, str) else purpose
        self._purpose = set(purpose)

    def __getattr__(self, attr):
        return getattr(self._cites, attr)

    def __getitem__(self, key):
        return getattr(self, key)
    
    @property
    def cites(self):
        """:class:`Citation` object (without purpose) of self"""
        return self._cites

    @property
    def purpose(self)->tuple[str]:
        """Purpose(s) for citation"""
        return tuple(self._purpose)

    def add_purpose(self, purpose:Union[str,Sequence[str]])->None:
        """
        Add purposes to purpose field of citation.

        Parameters
        ----------
        purpose : str | Sequence[str]]
            Purposes to add to citaion.

        """
        purpose = set() if purpose is None else purpose
        purpose = {purpose,} if isinstance(purpose, str) else purpose
        self._purpose |= set(purpose)


_tags:list[Citation] = list()
_citegroups:dict[str,list[Citation]] = dict()
_cited:list[CitedReason] = list()


def _get_citation_attr(check:str, attr:str)->Union[Citation,None]:
    """
    Get the :class:`Citation` that has a an attr matching name,
    if it exists, otherwise return None
    """
    for t in _tags:
        if hasattr(t, attr) and t[attr] == check:
            return t
    return None


def _get_citation_style(name:str, style:str)->Union[Citation,None]:
    """
    Get the :class:`Citation` that has a style kwarg matching name,
    if it exists, otherwise return None
    """
    for citation in _tags:
        if style == citation.get_style(name):
            return citation
    return None


def _update_citation(ct:Citation, tag:str=None, citation:str=None, 
                     overwrite=False, warn:bool=True, **kwargs)->None:
    """Add or modify a citation based on given values"""
    if tag is not None:
        ct.tag = tag
    if citation is not None:
        if overwrite:
            if warn:
                warnings.warn(f"replacing citation field of {ct}")
            ct.replace_citation(citation)
        else:
            ct.citation = citation
    ct.update_style(overwrite=overwrite, warn=warn, **kwargs)
        

def register_citation(tag:str=None, citation:str=None, overwrite:bool=False, 
                      warn:bool=True, **kwargs:str)->Citation:
    """
    Add a citation to the list of available citations. 
    
    This is the prefered way to add a new citation. 
    
    Note that this does not add the citation to the list
    returned when :func:`get_citations` is called, but rather makes it possible
    to specify a citation with :func:`cite` and :func:`add_citation` with just
    a tag name.

    Parameters
    ----------
    tag : str, optional
        Short identifier string used to identify citation internally. smfBursts
        citations follow the format ``[firstauthor][journalabreviation][year]``. 
        The default is None.
    citation : str, optional
        Full citation (smfBursts citations given in Nature format) string.
        The default is None.
    overwrite : bool, optional
        Allow overwriting of existing citation fields, if False, will raise an
        error. The default is False.
    warn : bool, optional
        Display warning if changing an already set field of an existing citation.
        The default is True.
    **kwargs : str
        Additional citation styles to include.

    Raises
    ------
    ValueError
        Multiple potential citations already registered that could be updated.

    Returns
    -------
    Citation
        :class:`Citation` object created or updated by the call.

    """
    ct = _get_citation_attr(tag, 'tag')
    cc = _get_citation_attr(citation, 'citation')
    if ct is None and cc is None:
        out = Citation(tag, citation, styles=kwargs)
        _tags.append(out)
        return out
    if ct is not None and cc is not None and ct is not cc:
        raise ValueError("tag and citation coorespond to different citations")
    out = cc if ct is None else ct
    _update_citation(out, tag=tag, citation=citation, overwrite=overwrite, warn=warn, **kwargs)
    return out


def create_citation_group(name:str, *tags:str, modify:bool=True, noreplace:bool=True)->None:
    """
    Create a citation-group name done by specifying a name for the group, and
    then the tags of all citations which should belone to said group.
    
    When :func:`cite` or :func:`add_citation` is called with ``name`` as an
    argument, all citations in group will be added to the citations list.

    Parameters
    ----------
    name : str
        name for new citation group
    *tags : list[str]
        tags of all citations in citation group.
    modify : bool, optional
        If True, then if a citation group already exists, add tags to group.
        If False, then replace tags
        The default is True.
    noreplace : bool, optional
        If modify is False, then if noreplace is True, raise an error when name
        already exists as a citation group. The default is True.

    Raises
    ------
    ValueError
        No tags specified or trying replace existing citation group.

    """
    name = str(name)
    if not tags:
        raise ValueError("must specify at least one tag")
    if noreplace and not modify and name in _citegroups:
        raise ValueError(f"cannot replace group {name}")
    citations = _citegroups[name] if modify and name in _citegroups else list()
    for tag in tags:
        tag = str(tag)
        ct = _get_citation_attr(tag, 'tag')
        if any(ct is ct for ct in citations):
            continue
        if ct is None:
            ct = register_citation(tag)
        citations.append(ct)
    _citegroups[name] = citations


def _citebyattr(names:list[str], attr:str)->list[Citation]:
    """
    Function for :func:`_get_cite` when using tags and cite_groups specification,
    for processing the tags kwarg
    """
    out = list()
    for name in names:
        ct = _get_citation_attr(name, attr)
        if ct is None:
            raise ValueError(f"unrecognized {attr}: {name}")
        if any(ct is c for c in out):
            continue
        out.append(ct)
    return out


def _citebygroup(groups:list[str])->list[Citation]:
    """
    Function for :func:`_get_cite` when using tags dn cite_groups specification,
    for processing the cite_groups kwarg
    """
    try:
        out = list(chain.from_iterable(_citegroups[gerr:=group] for group in groups))
    except KeyError:
        raise ValueError(f"unrecognized cite_group: {gerr}")
    return out


def _citebyargs(args:tuple[str,...])->list[Citation]:
    """Function for :func:`_get_cite` when using *args specification"""
    out = list()
    for arg in args:
        ct = _citegroups.get(arg, None)
        if ct is None:
            ct = _get_citation_attr(arg, 'tag')
            if ct is None:
                ct = register_citation(arg)
            ct = [ct, ]
        out.extend(ct)
    return out
        

def _cite_fromall(tags:list[str], citations:Union[None,list[str]], 
                  styles:Union[None,list[dict[str,str]]])->list[Citation]:
    """Function for :func:`_get_cite` when using tags, citations, styles format"""
    expect_len = len(tags)
    if citations is not None:
        citations = citations if isinstance(citations, (list, tuple)) else [citations,]
        if expect_len and expect_len != len(citations):
            raise ValueError("inconsistent number of tags and citations")
        else:
            expect_len = len(citations)
    if styles is not None:
        styles = styles if isinstance(styles, (list, tuple)) else [styles,]
        styles = [{str(k):v for k, v in style.items()} for style in styles]
        if expect_len and expect_len != len(styles):
            raise ValueError("inconsistent number of tags/citations and styles")
        else:
            expect_len = len(styles)
    if not expect_len:
        raise ValueError("must specify citations to include")
    if not tags:
        tags = repeat(None, expect_len)
    if not citations:
        citations = repeat(None, expect_len)
    if not styles:
        styles = repeat(dict(), expect_len)
    out = list()
    # check each citation for previously existing citations to potentially update
    for tag, citation, style in zip(tags, citations, styles):
        out.append(register_citation(tag, citation, **style))
    return out


def _extend_citation(citations:list[Citation], purpose:Union[str,Sequence[str]]=None)->None:
    """
    Internal function for :func:`cite` and :func:`add_citation` which incorporates
    each citation, and updates purpose to :attr:`_cited` list.
    """
    for ct in citations:
        must_add = True
        for c in _cited:
            if ct is c.cites:
                c.add_purpose(purpose)
                must_add = False
                break
        if must_add:
            _cited.append(CitedReason(ct, purpose))


def unique_citations(citations:list[Citation])->list[Citation]:
    """
    Remove duplicates from a list of :class:`Citation` objects.

    Parameters
    ----------
    citations : list[Citation]
        Input list of :class:`Citation` with potential repeats.

    Returns
    -------
    list[Citation]
        Input with repeats removed.

    """
    out = list()
    for citation in citations:
        if any(citation is c for c in out):
            continue
        out.append(citation)
    return out


def _get_cite(*args, tags:list[str]=None, cite_groups:list[str]=None, citations:list[str]=None, 
              styles:list[dict[str,str]]=None)->list[Citation]:
    """
    Internal function for :func:`cite` and :func:`add_citation` which processes input
    style, and calls appropriate function for updating :attr:`_tags` and :attr:`_cited`
    lists.
    """
    if args and (tags or cite_groups or citations or styles):
        raise ValueError("cannot specify args with other styles")
    if args:
        return unique_citations(_citebyargs(args))
    if tags is not None:
        tags = tags if isinstance(tags, (list, tuple)) else [tags,]
    else:
        tags = list()
    if cite_groups is not None:
        if citations or styles:
            raise ValueError("cannot mix cite_groups with citations or styles")      
        cite_groups = cite_groups if isinstance(cite_groups, (list, tuple)) else [cite_groups,]
        return unique_citations(_citebyattr(tags,'tag') + _citebygroup(cite_groups))
    return unique_citations(_cite_fromall(tags, citations, styles))


def cite(*args:str, tags:list[str]=None, cite_groups:list[str]=None, citations:list[str]=None, 
              styles:list[dict[str,str]]=None, purpose:Union[str,set[str]]=None)->Callable[[Callable],Callable]:
    r"""
    Decorator which adds a citation to citaion record, so that when
    :func:`get_citations` is called, the specified citation(s) appear in the
    citation list. This allows for functions to be tagged so that a citation is
    only added if they are called.
    
    There are three styles of calling this function
    
    #. \* args: specify cite-groups and tags together::
    
        @cite('tag1', 'tag2', 'group3', 'group4', purpose='method')
        def mymethod(*args, **kwargs):
            ...
    
    #. tag/cite-group keyword arguments, useful if tag and cite-group share a name::
    
        @cite(tags=['tag1', 'tag2'], cite_groups=['group3', 'group4'], purpose='method')
        def mymethod(*args, **kwargs):
            ...
    
    #. tags, citations, styles keyword arguments: specify new citations, completing each field::
    
        @cite(tags=['tag1', 'tag2'], citations=['citation1', 'citation2'],
                    styles=[{'bibtex':'bibtex1', 'ris':'ris1'},
                            {'bibtex':'bibtex2', 'ris':'ris2}], purpose='method')
        def mymethod(*args, **kwargs):
            ...
    
    Of these three, the the first is strongly prefered, where citations are registered beforehand with
    the :func:`register_citation` functions, and citation groups created beforehand with the
    :func:`create_citation_group` function.
    
    
    Parameters
    ----------
    *args : str
        cite-group and tag names of citations to be added. Note that cite-groups are searched
        first, so that if a cite-group and tag share the same name, the tag group citations
        will appear in the citation list, and not the tag citation.
    tags : list[str], optional
        tags (searches only tags) of each citation to add. Input as list of tags.
        If specified as single string, assumes a single tag is being added and converts
        string into ``[tag, ]`` list. If citations or styles are specified, length
        of tag list must match length of citations/styles lists.
        The default is None.
    cite_groups : list[str], optional
        Names of cite-groups to add to citations. Cite-groups must be specified in advance
        with :func:`create_citation_group`. The default is None.
    citations : list[str], optional
        String of citation as it would appear in a paper. Citations included in smfBursts use
        Nature format. The default is None.
    styles : list[dict[str,str]], optional
        List of dictionaries specifying format_type:citation key/value pairs. One for each
        citation. Citations included in smfBursts include bibtex, ris, and doi formats.
        The default is None.
    purpose : str | set[str], optional
        Purpose descriptor for all citations to add to notes when using :func:`get_citations`
        as ``smf.get_citations(include_purpose=True)``. Note that unlike all other keyword arguments,
        this argument does not create citations, and does not need to match the size of
        other specifications. The default is None.
    
    Returns
    -------
    (Callable[[Callable],Callable])
        wrapper function called on function being decorated.

    """
    cites = _get_cite(*args, tags=tags, cite_groups=cite_groups, citations=citations, styles=styles)
    called = False
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal called
            if not called:
                _extend_citation(cites, purpose)
                called = True
            return func(*args, **kwargs)
        return wrapper
    return decorator


def add_citation(*args, tags:list[str]=None, cite_groups:list[str]=None, citations:list[str]=None, 
              styles:list[dict[str,str]]=None, purpose:Union[str,set[str]]=None)->None:
    r"""
    Add a citation to citaion record, so that the specified citation(s) appear in the
    citation list.
    
    There are three styles of calling this function
    
    #. \* args: specify cite-groups and tags together::
    
        add_citation('tag1', 'tag2', 'group3', 'group4', purpose='method')
    
    #. tag/cite-group keyword arguments, useful if tag and cite-group share a name::
    
        add_citation(tags=['tag1', 'tag2'], cite_groups=['group3', 'group4'], purpose='method')
    
    #. tags, citations, styles keyword arguments: specify new citations, completing each field::
    
        add_citation(tags=['tag1', 'tag2'], citations=['citation1', 'citation2'],
                     styles=[{'bibtex':'bibtex1', 'ris':'ris1'},
                             {'bibtex':'bibtex2', 'ris':'ris2}], purpose='method')
    
    Of these three, the the first is strongly prefered, where citations are registered beforehand with
    the :func:`register_citation` functions, and citation groups created beforehand with the
    :func:`create_citation_group` function.

    Parameters
    ----------
    *args : str
        cite-group and tag names of citations to be added. Note that cite-groups are searched
        first, so that if a cite-group and tag share the same name, the tag group citations
        will appear in the citation list, and not the tag citation.
    tags : list[str], optional
        tags (searches only tags) of each citation to add. Input as list of tags.
        If specified as single string, assumes a single tag is being added and converts
        string into ``[tag, ]`` list. If citations or styles are specified, length
        of tag list must match length of citations/styles lists.
        The default is None.
    cite_groups : list[str], optional
        Names of cite-groups to add to citations. Cite-groups must be specified in advance
        with :func:`create_citation_group`. The default is None.
    citations : list[str], optional
        String of citation as it would appear in a paper. Citations included in smfBursts use
        Nature format. The default is None.
    styles : list[dict[str,str]], optional
        List of dictionaries specifying format_type:citation key/value pairs. One for each
        citation. Citations included in smfBursts include bibtex, ris, and doi formats.
        The default is None.
    purpose : str | set[str], optional
        Purpose descriptor for all citations to add to notes when using :func:`get_citations`
        as ``smf.get_citations(include_purpose=True)``. Note that unlike all other keyword arguments,
        this argument does not create citations, and does not need to match the size of
        other specifications. The default is None.

    
    """
    cites = _get_cite(*args, tags=tags, cite_groups=cite_groups, citations=citations, styles=styles)
    _extend_citation(cites, purpose)


def set_prefered_style(name:str)->None:
    """
    Set default style of citation to return when using :func:`get_citation`.

    Parameters
    ----------
    name : str
        Name of desired style, smfBursts citations always have ``'bibtex'``
        and ``'ris'`` styles.

    """
    Citation.set_prefered_style(name)


def list_citation_groups()->list[str]:
    """
    Get all currently existing citation-group names.

    Returns
    -------
    list[str]
        List of string identifiers for currently existing citation groups.

    """
    return list(_citegroups.keys())


def list_tags()->list[str]:
    """
    Get all currently registerd tag names.

    Returns
    -------
    list[str]
        List of string identifiers for currently registered tags.

    """
    return [tag.tag for tag in _tags if hasattr(_tags, 'tag')]


def registered_citations()->list[dict[str,str]]:
    """
    Get a list of dictionries indicating the full data of all currently registered
    citations.

    Returns
    -------
    list[dict[str,str]]
        List of dictionary representations of all currently registered citations.

    """
    return [tag.as_dict() for tag in _tags]


def registered_citation_groups()->dict[list[dict[str,str]]]:
    """
    Get a dictionary with keys of citation group names and keys as a list
    of dictionaries of each citation in citation group, dictionaries same
    format as in :func:`registered_citations`.

    Returns
    -------
    dict[list[dict[str,str]]]
        Dictionary of citation-group information.

    """
    return {group:[tag.as_dict() for tag in tags] for group, tags in _citegroups.items()}


def get_citations(style:str=None, include_purpose:bool=False)->list[Union[str,tuple[str,str]]]:
    """
    Get a list of all the citations that should be included in a paper that uses
    data generated by the curent script.

    Parameters
    ----------
    style : str, optional
        Style to retrieve of itation, usually either 'ris', or 'bibtex'. The default is None.
    include_purpose : bool, optional
        If ``False``, simply return the specified citation style. If `True``
        then each element of the list is a 2-tuple, the first element is the
        desired citation type, and the second a list of all the 'purposes' for
        which the given citation was used. (as specified in :func:`cite` or :func:`add_citation`)
        The default is False.

    Returns
    -------
    list[str | tuple[str, str]]
        All citations that should be included in a paper based on the current
        script, if ``include_purpose=True``, also includes purpose of each citation.

    """
    ip = not include_purpose
    if style is None:
        return [c.prefered_style for c in _cited] if ip else [(c.prefered_style, c.purpose) for c in _cited]
    if style == 'citation':
        return [c.citation for c in _cited] if ip else [(c.citation, c.purpose) for c in _cited]
    return [c.get_style(style) for c in _cited] if ip else [(c.get_style(style), c.purpose) for c in _cited]


def print_citations(linesize:int=80, print_string:bool=True, return_string:bool=False, alphabetize:bool=False)->None|str:
    citations = get_citations()
    if alphabetize:
        citations = sorted(citations)
    citations = (f'{i+1}. {citation}' for i, citation in enumerate(citations))
    citations = '\n'.join(chain.from_iterable((wrap(citation, linesize, 
                                                    subsequent_indent='    ') 
                                               for citation in citations)))
    if print_string:
        print(citations)
    return citations if return_string else None
    