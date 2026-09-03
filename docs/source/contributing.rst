Contributing
============

smfBursts is designed to be easily extended.
See the :ref:`defext`  user guide for creating your own extension module.

Stages of adding to smfBursts
-----------------------------

smfBursts is designed to be easily extended, and thus a platform to develop new methods.
Each method has a certain degree of maturity.

smfBursts will implement methods in different ways depending on the degree of maturity.

When a method is new, for instance the first implementation/publication,
new :class:`PhotonTable<smfbursts.photondata.PhotonTable>` classes should be placed
in a ``.py`` file that can be imported from the current directory.
This file should be made available on github or zenodo
that is linked in the supplementary data of the publication.
These classes need not implement all the special naming/defaults etc.
that are present in the mature methods that are included in the methods that
are implemented in the core of smfBursts.

As the data becomes more accepted, and the best defaults etc. are determiend,
new versions of the file should be added.
This represents the maturation of the method.
Additionally, greater though can be considered for the best implementation.

At this point, there is a choice to be made:

1. Publish the method as a separte package uploaded to PyPi/conda-forge
2. Incorporate the method into the next version of smfBursts

It is appreciated at this point to reach out to the maintainer of smfBurst, 
currently Paul Harris, to discuss the best options.

Generally methods that have been shown to be 

- Significant advancement
- used by more than 1 research group (indicating the method is accepted by the community)

will be considered.
The choice of a separate package or direct inclusion will be based on how complicated the implementation is.
If the method consists of a single new :class:`PhotonTable<smfbursts.photondata.PhotonTable>`
or a new column within one of those classes, 
the method will probably be directly integrated into the next version of smfBursts.
On the other hand, if proper implementation means a larger structure with
additional functions etc. to compute derived values etc., then a separate python package
will likely be recomended.

To maintain consistency, I encourage the package name to end in "bursts", e.g. H2MMbursts.

Finally, whether implemented as a separte package or into smfBursts directly,
new methods should have the following implemented:

- appropriate citation
- Column naming function/string


Citation Request Policy
-----------------------

The citations module is designed to help both users know who to cite,
and method developers to get their citations.

But there need to be limits, not every paper using a burst search algorithm should be cited
when that algorithm is used.

So the basic policy for smfBursts is simple: what is the minimum reasonable citation?

So for a given method, the citation(s) it should generate with the ``get_citations`` function
should be the citation(s) that introduced the method in the form it was used.

Papers that contributed to the development of said method should not be included.
Unless they would also be cited in an article using the method.

I (Paul) am not perfect, it is likely I missed some citations in the methods, therefore,
if you feel like a particular is implemented in smfBursts, but doesn't produce a citation,
please reach out and we can work together to get your citation added.
But, be reasonable, I can't be inundated with all requrests.

