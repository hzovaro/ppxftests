PROBLEM: 

In the process of making figures for the paper showing fits to individual galaxies, it's become clear that we have 2 problems:

1) the fit itself is not very good for a handful of galaxies, and
2) the CLEAN keyword is masking out a *lot* of pixels in some galaxies (although by eye this doesn't appear to affect the fit all that much??)

TODO:
1. In the documentation, it says only to use the CLEAN keyword if the reduced-chi2 of the fit is ~1. So, for each galaxy, check pp.chi2. If they are all >1, then it's probably best to repeat everything with CLEAN turned off.
    A: chi2 > 1 for the vast majority of galaxies. Probably best to re-run with CLEAN turned off. 
2. How does the chi2 change from MC run to MC run? Is there a lot of variation? Test: run ~20 iterations for say ~3 galaxies & look at the MC plots.
    A: it doesn't change by all that much over 100 runs (width of distribution < 1)
3. Re-run w/ clean turned off.
    3a. run in debug mode to double-check everything looks OK.
    3b. change directory structure to preserve fits and figures with clean turned on.
        Try plotting figures w/ clean turned on to check paths, etc. work.
    3c. re-run on avatar.
        First just run the RE1 ones to see how long they take (& if any bugs arise).
        Then queue the others.
5. Once fits are finalised, look at the ppxf fits & make a list of galaxies with poor fits:
    
    (These are for the RE1 aperture)
    ESO362-G18 - 29.51 (tentative - residuals) - unimodal chi2 dist.
    FAIRALL49 - 12.49 (really bad - residuals) - unimodal chi2 dist.
    IC4329A - 112.97 (really bad - residuals) - unimodal chi2 dist.
    MCG-03-34-064 (really bad - residuals) - bimodal chi2 dist.
    NGC424 - 19.32 (tentative - residuals) - unimodal chi2 dist.
    NGC1667 - 30.54 (really severe wavelength mismatch between blue & red) - unimodal chi2 dist.
    NGC5506 - 5.60 (tentative - quite noisy) - bimodal chi2 dist.
    NGC6860 - 10.22 (really bad - unclear why) - unimodal chi2 dist.
    NGC7469 - 466.68 (really bad - residuals) - unimodal chi2 dist (but quite messy)

6. Try masking ± 150 Angstroms around Halpha and [OIII], just in a single ppxf run.

    Results:
    for MCG-03-34-064, whether or not we get a good fit seems to be purely random - you can run the code multiple times and get different fits...
    When it's bad, the reduced-chi2 is ~44, when it's good it's ~30. What does the hist look like?
    It is strongly bimodal - a small peak at 30 and a much larger one at 50.
    Same situation for NGC5506, but not for the other galaxies with poor fits.

7. Options:
    1. Find a robust way to ensure a good fit for all of these galaxies, or 
    2. Exclude them from the analysis.


Exploring option 1:

    ESO362-G18 - 29.51 (tentative - residuals) - unimodal chi2 dist.
    NGC424 - 19.32 (tentative - residuals) - unimodal chi2 dist.

    Done:
        MCG-03-34-064
        FAIRALL49
        NGC6860
        NGC424
        NGC1667

    Galaxies to abandon:
        NGC5506 - really noisy... better just stick with current results 
        IC4329A? - really hard to fine tune - abs. features are shallow... 
        NGC7469 - simply cannot get fit to work...

8. Inspecting the fits after masking out additional wavelength regions:

    ESO362-G18 - good
    FAIRALL49 - good
    MCG-03-34-064 - good 
    NGC424 - good
    NGC1667 - good
    NGC5506 - still noisy but has high x_AGN so probably won't be used anyway?
    NGC6860 - good
    
    IC4329A - BAD
    NGC7469 - BAD
        
+-----------------------------------------------------------------+
        
A note on indexing and age cutoffs:
    in compute_lw_age():
        input age_thresh_upper (= tau_cutoff)
        LW ages computed in templates up to but NOT including tau_cutoff.
        so if tau_cutoff = ages[idx] the LW age is computed from the templates from ages[0] to ages[idx - 1]

        so in the array of LW ages:
            idx 0 --> tau_cutoff idx = 1
            idx 1 --> tau_cutoff idx = 2
            ...
            idx N --> tau_cutoff idx = N + 1
            so for tau_cutoff = 100 Myr,
            idx <100 Myr idx - 1> --> tau_cutoff idx = <100 Myr idx>.
