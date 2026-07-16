jazz.00054 REPAIR — note (the repaired files live in ../raw/)

We did NOT build a separate fixed/ tree. The only data defects worth repairing
were one corrupt track and its missing grey spectrogram, so the two repaired
files were patched directly into raw/ (the path the data loader reads):

  raw/genres_original/jazz/jazz.00054.wav    known-good replacement (27.38s clip)
  raw/images_grey_scale/jazz/jazz00054.png   128x128 L mel spectrogram, made by
                                             src/data_doctor.py and histogram-
                                             matched to the existing jazz greys
                                             (mean ~116; corpus 120 +/- 5)

CONSEQUENCE: raw/ is no longer byte-for-byte the original GTZAN — it was patched
in place for loader simplicity (one read path, no branching). The pre-fix state
(corrupt wav, jazz grey short by one) is recorded only in before/eda_stats.json.
This after/ folder holds the post-fix EDA snapshot (eda_stats.json, produced by
`python eda/run_eda.py --phase after --data-root projects/genre/data/raw`), which
should show jazz whole and the grey representation at 1000/1000.

NOT altered (documented only): 10 short 3-sec tracks (9 segments), 10 long
hiphop clips (30.649s). Honest properties of the data — see ../README.md.
