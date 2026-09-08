# Review evaluation

Each cohort is evaluated separately. For N patients and budget b, k = floor(b*N) patients are reviewed. Risk is mean absolute RV EF error over the N-k retained patients. Review does not imply that deferred cases were corrected.

The supported protocol retains budgets 0.10 and 0.20 and 5,000 patient bootstrap draws. All models, scores, budgets, and policies share each cohort's draws, including repeated patient occurrences. Summaries use percentile intervals and medians across the actual supplied models.

The five policies are random, QC alone, plausibility plus random, plausibility plus QC, and an unconstrained absolute-reference-EF-error oracle. Random allocations use analytical expected risks. The oracle requires references and is not a deployable policy.

Flags use predicted EDV <= 0, predicted ESV > EDV, predicted EF outside [0,100], or nonfinite predicted EF. Finite out-of-range EF is not clipped. The external adapter explicitly rejects undefined/nonfinite inputs unsupported by its labelled audit. Flags stay in N and consume the same k places.

QC order is descending score then ascending patient ID. Finite flagged cases use the original ratio-severity ordering before the patient-ID tiebreaker. Prediction-only selections do not consult references. See INPUT_CONTRACT.md for the full validation rules.

Within each model:

```
Delta1 = risk(plausibility + random) - risk(plausibility + QC)
Delta2 = risk(QC alone) - risk(plausibility + random)
```

Positive Delta1 favors adding QC after plausibility screening. Negative Delta2 favors QC alone over plausibility plus random. Compute contrasts within model before taking their median; subtracting separately summarized risks can give a different result. Intervals are exploratory and unadjusted. Architectures are fixed models, not independent patient samples.
