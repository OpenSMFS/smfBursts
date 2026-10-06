Column Index
============

This page provides a list of all columns provided in by the ``smfbursts`` module.
Note that columns are grouped by category, and therefore may not have the same |Param|

This is inteded to serve as a quick reference, check here to see if a column that implements
the measure you are interested in is present, and follow the direction accordingly.

In keys, if an = sign is present, that key can be skipped, and the value after the
sign indicates the default value.

Intensity and Ratio Column
--------------------------

These are columns that rely on the total photon counts in a burst,
either returning the counts, or ratios thereof.

This type of column is generally use for gating and in standard FRET analysis.

nph_raw
*******

    |Param| : |BasePhotonTable|, typcially |Bursts| or |BurstOvlp|

    Type : :code:`np.int64`

    Keys : (``phsel`` : |PhSel| , )

The raw photon counts of the stream specified by ``phsel``


nph_bg
******

    |Param| : |NphBG|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The background corrected photon counts in the stream specified by ``phsel``.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.


nph_c
*****

    |Param| : |Ratios|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The "fully" corrected photon counts in the stream specified by ``phsel``.
This means both background correction and correction by the ``corr_mat`` matrix.
Depending on the level of correction in ``corr_mat`` are typically either cross-talk correction,
or cross-talk and cross-section based values.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.

ratio_raw
*********

    |Param| : |BasePhotonTable|, typcially |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel_num`` : |PhSel|, ``phsel_dem`` : |PhSel|)

The ratio of raw photon counts (from nph_raw_ ) between photon selections.
Evaluates as ``phsel_num / phsel_dem``.


E_raw
.....

    Remapping of ratio_raw_ .

    Keys : ()

Raw transfer efficiency for 2 emission experiment.

This is equivalent to

``Column(basephoton_param, 'ratio_raw', (Phsel('0ex0em'), PhSel('0ex')))``

S_raw
.....

    Remapping of ratio_raw_ .

    Keys : ()

Raw stoichimetry for 2 excitation 2 emission experiment.
This is equivalent to

``Column(basephoton_param, 'ratio_raw', (Phsel('0ex'), PhSel('0ex_1ex1em')))``

ratio_bg
********

    |Param| : |NphBG|

    Type : :code:`np.float64`

    Keys : (``phsel_num`` : |PhSel|, ``phsel_dem`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The ratio of background corrected photon counts (from nph_bg_ ) between photon selections.
Evaluates as ``phsel_num / phsel_dem``.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.

E_bg
....

    Remapping of ``ratio_bg``

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

Background corrected transfer efficiency for 2 emission experiment.

This is equivalent to

``Column(nph_param, 'ratio_bg', (PhSel('0ex0em'), PhSel('0ex'), startime, istoptime))``

S_bg
....

    Remapping of ``ratio_bg``

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

Background corrected stoichimetry for 2 excitation 2 emission experiment.

This is equivalent to

``Column(nph_param, 'ratio_bg', (PhSel('0ex'), PhSel('0ex_1ex1em'), startime, stoptime))``


ratio_c
*******

    |Param| : |NphBG|

    Type : :code:`np.float64`

    Keys : (``phsel_num`` : |PhSel|, ``phsel_dem`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The ratio of "fully" corrected photon counts (from nph_c_ ) between photon selections.
Level of correction is defined by correction in ``corr_mat``.

Evaluates as ``phsel_num / phsel_dem``.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.


E
....

    Remapping of ratio_c_

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

``corr_mat`` and background corrected transfer efficiency.
In a standard FRET experiment, if ``corr_mat`` accounts for :math:`\alpha, \delta, \gamma, \beta` this
can be considered the FRET efficiency.

This is equivalent to

``Column(ratio_param, 'ratio_c',(PhSel('0ex0em'), PhSel('0ex'), startime, stoptime))``


S
....

    Remapping of ratio_c_

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)


``corr_mat`` and background corrected stoichiometry.
In a standard FRET experiment, if ``corr_mat`` accounts for :math:`\alpha, \delta, \gamma, \beta` this
can be considered the fully corrected stoichiometry.

This is equivalent to

``Column(ratio_param, 'ratio_c',(PhSel('0ex'), PhSel('0ex_1ex1em'), starttime, stoptime))``


anisotropy_raw
**************

    |Param| : |BasePhotonTable|, typcially |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel_p`` : |PhSel|, ``phsel_s`` : |PhSel|)

The "anisotropy" raw photon counts (from nph_raw_ )
of streams ``phsel_p`` and ``phsel_s``
specifying parallel and perpendicular channels respectively.
Evaluates as ``(phsel_p - phsel_s) / (phsel_p + 2*phsel_s)``.

.. note::

    The intended use of this column is for ``phsel_p`` to specify the parallel
    channel, and ``phsel_s`` to specify the perpendicular channel.
    Depending on the ``rcPrams['warn.anisotropy`]`` value,
    specifying channels that do not mean this criterion may

        - raise an error, (``"raise"``)
        - cause a warning (``"warn"``, default when loading smfBursts)
        - nothing (``"ignore"``)


anisotropy_bg
*************

    |Param| : |NphBG|

    Type : :code:`np.float64`

    Keys : (``phsel_p`` : |PhSel|, ``phsel_s`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The "anisotropy" of background corrected photon counts (from nph_bg_
of streams ``phsel_p`` and ``phsel_s``
specifying parallel and perpendicular channels respectively.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.

.. note::

    The intended use of this column is for ``phsel_p`` to specify the parallel
    channel, and ``phsel_s`` to specify the perpendicular channel.
    Depending on the ``rcPrams['warn.anisotropy`]`` value,
    specifying channels that do not mean this criterion may

        - raise an error, (``"raise"``)
        - cause a warning (``"warn"``, default when loading smfBursts)
        - nothing (``"ignore"``)


anisotropy_c
************

    |Param| : |Ratios|

    Type : :code:`np.float64`

    Keys : (``phsel_p`` : |PhSel|, ``phsel_s`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The anisotropy of corrected photon counts (from nph_c_ ) of streams ``phsel_p`` and ``phsel_s``
specifying parallel and perpendicular channels respectively.

Correction carried out by ``corr_mat`` for correction of intensities,
and if specified, ``l1`` and ``l2`` (parameters of |Ratios| correct for
mixing of polarizations due to objective (primarily from NA), as defined by
|Koshioka| and |Schaffer|.

The equation given in |Schaffer| is

.. math::

    r = \frac{_{corr}n_{\parallel} - _{corr}n_{\perp}}{(1-3l_{2}) _{corr}n_{\parallel} - (2-3l_{1}) _{corr}n_{\perp}}


If ``l1`` and ``l2`` are not specified, then they are treated as :math:`0`,
The lack of the factor :math:`G` reported in |Schaffer| is because ``corr_mat`` is
expected to account for this factor (in the diagonal),
*i.e.* :math:`_{corr}n_{\parallel} = Gc_{x}`

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.

.. note::

    The intended use of this column is for ``phsel_p`` to specify the parallel
    channel, and ``phsel_s`` to specify the perpendicular channel.
    Depending on the ``rcPrams['warn.anisotropy`]`` value,
    specifying channels that do not mean this criterion may

        - raise an error, (``"raise"``)
        - cause a warning (``"warn"``, default when loading smfBursts)
        - nothing (``"ignore"``)



Temporal (Macrotime) Based Columns
----------------------------------

These columns are columns that inspect the macrotimes of bursts,
but not concenred with overall intensities.
These define the limits of bursts, and the distribution of photons within bursts.

start
*****

    |Param| : |BasePhotonTable|

    Type: :code:`np.int64`

    Keys : ()

Start time of burst in ``clk_p`` units.

.. note::

    **For Developers**

    When defining a new |BasePhotonTable| that must be accesible
    at the end of ``__init_columns__``


stop
****

    |Param| : |BasePhotonTable|

    Type: :code:`np.int64`

    Keys : ()

Stop time of burst in ``clk_p`` units.

.. note::

    **For Developers**

    When defining a new |BasePhotonTable| that must be accesible
    at the end of ``__init_columns__``



istart
******

    |Param| : |BasePhotonTable|

    Type: :code:`np.int64`

    Keys : ()

Index of first photon in ``photondata.times`` in burst.

.. note::

    **For Developers**

    When defining a new |BasePhotonTable| that must be accesible
    at the end of ``__init_columns__``



istop
*****

    |Param| : |BasePhotonTable|

    Type: :code:`np.int64`

    Keys : ()

Index of end of burst, this is one greater thatn the index of the last photon in burst.
This allows ``photondata.times[start:stop]`` to be used to select all photons in burst.

.. note::

    **For Developers**

    When defining a new |BasePhotonTable| that must be accesible
    at the end of ``__init_columns__``



istarttime
**********

    |Param| : |BasePhotonTable|

    Type : :code:`np.int64`

    Keys : ()

Time in ``clk_p`` units of first photon in burst,
(this is equivalent to the start of bursts in most previous burst search definitions).


istoptime
*********

    |Param| : |BasePhotonTable|

    Type : :code:`np.int64`

    Keys : ()

One greater than the time in ``clk_p`` units of last photon in burst,
(this is equivalent to the stop of bursts in most previous burst search definitions).
The ``time+1`` convention allows burst duration to be calculated as ``istoptime - istarttime``.


dur
****

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The duration of each burst, in seconds.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.


sep
****
**Non-Atomic Column**

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

Separation between bursts in seconds. This is non-atomic column with is size 1 smaller than size of table.

This is used in recurrence analysis.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the points where bursts are considered to begin and end.
The options and default are specified by the subclass of |BasePhotonTable|.


midtime
*******

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColyKeyStop = *subclass default*)

The midpoint (mean) of the start and stop time of bursts. This can be considered the
"actual" time of the burst.

``starttime`` and ``stoptime`` are used to define the start and stop times of
that determine the duration of the period for which the background counts are evaluated.
The options and default are specified by the subclass of |BasePhotonTable|.


meanT
*****

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel| , )

Mean time of photons in ``phsel`` of burst, in seconds.
This is a centroid of the burst, with primarily photophysical
transitions biasing towards beginning or end
depending on the stream and dynamics occuring.


mTdiff
******

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel_a`` : |PhSel| , ``phsel_b`` : |PhSel|)

The differnce between the means (as evaluated by the meanT_ column)
of ``phsel_a`` and ``phsel_b``, evaluated as ``phsel_a - phsel_b``.

This is more usefull for determing if dynamics are occuring compared to meanT_ .

If for example, there is bleaching of the acceptor during the burst,
then if ``phsel_a`` is the donor stream and ``phsel_b`` is the acceptor stream,
then mTdiff_ will be more possitive.
As the acceptor photons will be more intense earlier in the burst,
biasing the ``meanT`` of ``phsel_b`` toward shorter times.


Rate Based Columns
------------------

These columns report values around photon rates, taking both photon counts
and time into account.

max_rate
********

    |Param| : |BasePhotonTable|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel| , ``m`` : :code:`int` = 10)


Maximum (peak) photon rate in stream ``phsel`` using a sliding window of size ``m``.
Units of :math:`photons\:s^{-1}`.


bg
****

    |Param| : |BG|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel| , )

Background count rate of photons in stream ``phsel``.
Units of :math:`photons\:s^{-1}`.


bg_err_CM
*********

    |Param| : |BG|

    Type : :code:`np.float64`

    Keys : (``ph_sel`` : |PhSel|, )


Crames-von Mises error metric.
Computes :math:`\int_{-\infty}^{\infty} \left[] L_{n} - L_{*} \right]^{2}`.
Using trapezoid rule for numerical integration.


bg_err_KS
*********

    |Param| : |BG|

    Type : :code:`np.float64`

    Keys : (``ph_sel`` : |PhSel|, )


Kolmogorov-Smirnov error metric, computes the error as the max of deviation of
the empirical CDF from the fitted CDF.


range_counts
************

**mapping from one table to another**

    Source |Param| : |BG|

    Destination |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

The expected number of background photons in the stream ``phsel`` in the time range
defined by ``starttime`` and ``stoptime``.

This is simply the multiplication of the background count rate (``bg`` column of source param)
for the period in which ``starttime`` and ``stoptime`` belong (from destination param),
multiplied by the duration of time range of the destination param.

sbr
****

    |Param| : |NphBG|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel| , ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

Signal to background ratio in stream ``phsel``.
This is computed using the ``nph_raw`` column of the ``base`` parent of |NphBG|
and the ``rangecounts`` column of the ``bg`` parent of |NphBG|,
and computed ``nph_raw / rangecounts``.

``starttime`` and ``stoptime`` are used to defined the start and stop of the
period to identify the exact duration for which the background is evaluated.
The options and default are specified by the subclass of |BasePhotonTable| that is the
base param of the |NphBG| table of the source param.


Nanotime Based Columns
----------------------

These are columns used only when pulsed lasers are present.
These are based on fluoresence decays, closly related to fluoresence lifetime.

nanomean
********
*Discouraged, use nanomean_bg_ instead.*

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, )


Mean nanotime in ``phsel``.

.. note::

    It is better to use nanomean_bg_ (the next column),
    to assess the approximate lifetime of the burst, as nanomean_bg_
    accounts for the effect of background, which tends to make
    ``nanomean`` longer than the actual lifetime.


nanomean_bg
***********

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

Mean nanotime with background correction. 
This is nearly as accurate as maximum likelihood estimation based methods for determing
the mean fluoresence decay lifetime.

The expected number of background photons in the stream ``phsel`` in the time range
defined by ``starttime`` and ``stoptime``.


Correction factor is as follows

.. math::

    \tau = \frac{ \left(\displaystyle\sum_{i=1}^{N}{t_{i}}\right) - n_{bg}\bar{t_{bg}}}{N-n_{bg}}

Where :math:`\bar{t_{bg}` is the mean time of the excitation window,
and  thus the expected mean value of the background,
:math:`n_{bg}` is the expected number of background photons
(this is the ``'rangecounts'`` colmn of the ``'bg'`` parent of |NphBG|),
:math:`N` is the total number of photons in the burst, and
:math:`t_{i}` is the delay time from the laser pulse of the :math:`i^{th}` photon.

Both :math:`t_{i}` and :math:`\bar{t_{bg}}` are in units of seconds,
and time 0 is set by the method specified in ``irfstyle``.
The options for ``irfstyle`` are as follows:
if ``'thresh'``, set time 0 as ``irf_thresh``, 
if ``'mean'`` then set time 0 as mean of IRF,
if ``'max'`` then set time 0 as time of maximum value of IRF.


nmdiff
******

*Discouraged, use nmdiff_bg_ instead.*

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel_a`` : |PhSel|, ``phsel_b`` : |PhSel|, )

Difference in mean nanotimes (using nanomean_ ) between ``phsel_a`` and ``phsel_b``.
Assessed as ``phsel_a - phsel_b`` so if ``phsel_a`` has a longer
mean nanotime, then the value will be positive.

This can be applied in several ways to assess the coupling between fluorophores,
though in standard FRET experiments this tends to carry very little information.

.. note::

    It is better to use nmdiff_bg_
    to assess the approximate lifetime of the burst, as nmdiff_bg_
    accounts for the effect of background.


nmdiff_bg
*********

    |Param| : |NphBG|

    Type : :code:`np.float64`

    Keys : (``phsel_a`` : |PhSel|, ``phsel_b`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

Difference in background corrected mean nanotimes of streams ``phsel_a`` and ``phsel_b``.
Assesed as ``phsel_a - phsel_b``so if ``phsel_a`` has a longer mean nanotime,
then the value will be positive.

This can be applied in several ways to assess the coupling between fluorophores,
though in standard FRET experiments this tends to carry very little information.

The expected number of background photons in the stream ``phsel`` in the time range
defined by ``starttime`` and ``stoptime``.


rotational_corr_c
*****************

    |Param| : |Ratios|

    Type : :code:`np.float64`

    Keys : (``phsel_a`` : |PhSel|, ``phsel_b`` : |PhSel|, ``starttime`` : ColKeyStart = *subclass default*, ``stoptime`` : ColKeyStop = *subclass default*)

An estimate of the rotational correlation time using the nanomean_bg_ and
anisotropy_bg_ columns. Assuming a single anisotropy decay and single
fluoresence lifetime component.

The correlation time is calculated as follows

.. math::

    \rho = \frac{r\bar{\tau}}{r_{\circ} - r}


Where :math:`r_{\circ}` is the anisotropy at zero, specified by ``r0``,
and :math:`r` is the observed
anisotropy, specifically the anisotropy_c_ column,
which defaults to :math:`0.4`,
and :math:`\bar{\tau}` is the
fluoresence lifetime,
calculated as

.. math::

    \bar{tau} =
    \frac{n_{bg\parallel}\bar{\tau_{bg\parallel}}} +
    2 n_{bg\perp}\bar{\tau_{bg\parallel}}
    {n_{bg\parallel}+2 n_{bg\perp}}

where :math:`\n_{bg\parallel}` and :math:`\n_{bg\perp}` are the values
of the corrected photon counts (from nph_c_ )
of ``phsel_p`` and ``phsel_s`` respectively,
and :math:`\bar{\tau_{bg\parallel}}` and :math:`\bar{\tau_{bg\perp}}`
are the background adjusted mean nanotimes (from nanomean_bg_ ) of
``phsel_p`` and ``phsel_s`` respectively.

``starttime`` and ``stoptime`` define the starting and ending times
of the periods to compute the background counts.
The options and default is defined by the |BasePhotonTable| subclass.


.. warning::

    This column should been seen as highly approximate, as the rotational
    corrleation typically requires a large number of photons for accurate
    computation.

    Since this relies on the background adjusted nanomean (nanomean_bg_ ),
    as opposed to properly fitting the respective decays,
    and relies on a simplified version of the conversion from lifetime to
    anisotropy, this column is not a proper caclulation of the rotational
    correlation.

    Proper computation of lifetimes and rotational correlation should be done
    on ensembles of bursts belonging to a single population,
    with the decays collected.


phasor_g
********

    |Param| : |Phasor|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, irfstyle : {'thresh', 'mean', 'max'} = 'mean')

:math:`g` phasor value according to |Digman|.
Used in congunction with ``phasor_s`` to assess if there are multiple components
to the fluoresence decay.

phasor_s
********

    |Param| : |Phasor|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel`` : |PhSel|, irfstyle : {'thresh', 'mean', 'max'} = 'mean')

:math:`s` phasor value according to |Digman|.
Used in congunction with ``phasor_g`` to assess if there are multiple components
to the fluoresence decay.

nanohist
********

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.int64` 2D array

    Keys : (``phsel`` : |PhSel|, full : :code:`bool` = False )

The counts of number of photons in the stream ``phsel`` in each time range in each tcspc bin.
Returned array is 2D, ``[time_range, tcspc_bin]``.

``full`` is an option for how to use excitation range.
If :code:`False`, then the 0 index in the retunred array cooresponds to the beginning of the excitation range,
and the second dimension of the returned array is the size of the excitation window.
If :code:`True`, then the indexes in the returned array match those of nanotimes (original tcspc bin),
and the second dimension of the returned array is the size of the total number of tcspc_bins of the original data,
and indexes outside of the excitation range of ``phsel`` have a value of 0.


Dynamics Columns
----------------

Use these columns to idenify if within burst dynamics are occuring.

bva
****

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.float64`

    Keys : (``phsel_num`` : |PhSel|, ``phsel_dem`` : |PhSel|, n : :code:`int` = 10)

Standard deviation of the ratio of the number of photons in ``phsel_num`` and ``phsel_dem``
in chuncks of size ``n``. ``phsel_num`` must be entirely contained within ``phsel_dem``.
This was proposed in |Torella|.

When plotted against the equivalent ``ratio_raw``,
this can be used to assess a time range for dynamics in the given ratio.


ebva
****

    |Param| : |BasePhotonTable|

    Type : :code:`np.float64`

    Keys : (``phsel_num`` : |PhSel|, ``phsel_dem`` : |PhSel|, n : :code:`int` = 10)

"Excess" standard deviation of the ratio of the number of photons in ``phsel_num`` and ``phsel_dem``
in chuncks of size ``n``. ``phsel_num`` must be entirely contained within ``phsel_dem``.
This was proposed in |Terterov|.

.. math::

    S^{2} = s^{2} - \sigma^{2}

where :math:`s^{2}` is the square of the ``bva``, and

.. math::

    \sigma^{2} = \frac{\left<\epsilon\right>(1-\left<\epsilon\right>)}{n}

where :math:`\left<\epsilon\right>` is the mean ratio of the photon streams.
This therefore is simply the ``ratio_raw`` column.


fret2cde
********

    |Param| : |KDE|

    Type : :code:`np.float64`

    Keys : (``phsel_d`` : |PhSel|, ``phsel_a``: |PhSel|)

The FRET 2CDE value proposed in |Tomov| of ``phsel_d`` and ``phsel_a``.

.. math::

        FRET-2CDE \left( t_{CHD}, t_{CHA} \right) = 110 - 100 \cdot \left[ (E)_D + (1 - E)_A \right]


alex2cde
********

    |Param| : |KDE|

    Type : :code:`np.float64`

    Keys : (``phsel_d`` : |PhSel|, ``phsel_a``: |PhSel|)

The ALEX 2CDE value proposed in |Tomov| of ``phsel_d`` and ``phsel_a``.

.. math::

        ALEX-2CDE \left( t_{CHD}, t_{CHA} \right) = 
        100 - 50 \left[ BR_{D_{EX}} + BR_{A_{EX}} \right]



Photon Array Columns
--------------------

Photon array columns have rows that are themselves variable length numpy arrays.
If using :meth:`Photondata.iter_column() <smfbursts.photondata.PhotonData.get_column>` 
the return value will be a numpy object array,
while when using :meth:`Photondata.iter_column() <smfbursts.photondata.PhotonData.iter_column>`
each iteration returns a numpy array.

The array for each row is an array of the photons 

These columns are rarely used by the end user, rather they are most useful
when implementing new methods, allowing 

ph_times
********

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.ndarray[np.int64]`

    Keys : ( ``phsel`` : |PhSel|, )

Arrays of the photon macrotimes of photons in ``phsel``.

ph_dets
*******

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.ndarray[np.uint8]`

    Keys : ( ``phsel`` : |PhSel|, )

Arrays of the detector indecies of photons in ``phsel``.

ph_nanos
********

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.ndarray[np.uint16]`

    Keys : ( ``phsel`` : |PhSel|, )

Arrays of the photon nanotimes (not shifted for excitation window) of photons in ``phsel``.


ph_particles
************

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.ndarray[np.uint8]`

    Keys : ( ``phsel`` : |PhSel|, )

Arrays of the photon particle index of photons in ``phsel`` (simulated data only).


ph_mask
********

    |Param| : |BasePhotonTable|, typically |Bursts| or |BurstOvlp|

    Type : :code:`np.ndarray[np.bool\_]`

    Keys : ( ``phsel`` : |PhSel|, )

Boolean arrays of the all photons in time ranges, :code:`True` where a photon is in ``phsel``.



.. |Param| replace:: :class:`Param <smfbursts.datamodel.tables.Param>`
.. |PhSel| replace:: :class:`PhSel <smfbursts.ph_sel.PhSel>`
.. |BasePhotonTable| replace:: :class:`BasePhotonTable <smfbursts.photondata.BasePhotonTable>`
.. |Periods| replace:: :class:`Periods <smfbursts.backgroundtables.Periods>`
.. |BG| replace:: :class:`BB <smfbursts.backgroundtables.BG>`
.. |Bursts| replace:: :class:`Bursts <smfbursts.childphotontables.Bursts>`
.. |BurstOvlp| replace:: :class:`BurstOvlp <smfbursts.childphotontables.BurstOvlp>`
.. |NphBG| replace:: :class:`NphBG <smfbursts.childphotontables.NphBG>`
.. |Ratios| replace:: :class:`Ratios <smfbursts.childphotontables.Ratios>`
.. |KDE| replace:: :class:`KDE <smfbursts.childphotontables.KDE>`
.. |Phasor| replace:: :class:`Phasor <smfbursts.childphotontables.Phasor>`
.. |Digman| replace:: `Digman et. al. 2008 <https://doi.org/10.1529/biophysj.107.120154>`__
.. |Koshioka| replace:: `Koshioka, Sasaki, Masuhara 1995. <https://doi.org/10.1366/0003702953963652>`__
.. |Schaffer| replace:: `Schaffer et. al. 1999. <https://doi.org/10.1021/jp9833597>`__
.. |Terterov| replace:: `Terterov et. al. <https://doi.org/10.1016/j.bpr.2023.100116>`__
.. |Tomov| replace:: `Tomov 2012 <https://doi.org/10.1016/j.bpj.2011.11.4025>`__
.. |Torella| replace:: `Torella et. al. 2011 <https://doi.org/10.1016/j.bpj.2011.01.066>`__
