#!/bin/bash
# Regenerates every result, table and figure. Usage: bash run_all.sh /path/to/TectonicSetting_1.xlsx
# Each step is resumable; total run time is several hours on two cores.
set -e
D="$1"; mkdir -p res/oof cache figs
python3 run_baselines.py "$D"     # 13 classifiers + majority reference, both protocols; fills cache/ (fold-internal features) and res/oof/
python3 leak.py "$D"              # nearest-neighbour / shared-code statistics (res/leak.json)
python3 revision_exp.py "$D"      # merged-code partition, immobile-element set, MgO window, per-code table (res/revision.json)
python3 rep_grouped.py "$D"       # five repetitions with random assignment of codes to folds (res/repeated_grouped.json)
python3 perm_coherent.py "$D"     # permutation importance with ratios recomputed (res/perm_coherent.csv)
python3 dups.py "$D"              # near-duplicate analyses (res/dups.json, res/dup_pairs.npy)
python3 dedup.py "$D"             # grouped validation after removal of near-duplicates (res/dedup.json)
python3 exp_train.py "$D"         # augmentation schemes and controls, coarse-label training, LightGBM configurations and seeds (res/exp.jsonl)
python3 ternary.py "$D"           # nested ternary search, three- and four-group schemes (res/ternary.json)
python3 ternary2.py "$D"          # two-step diagrams, arc against all other settings (res/ternary2.json)
python3 ternary_oceanic.py "$D"   # oceanic diagram: nested search, fixed Nb-La-Yb, comparators (res/ternary_oceanic.json)
python3 analysis.py "$D"          # all pooled statistics and bootstrap intervals (res/V.json)
python3 figs.py "$D"; python3 fig7.py "$D"; python3 extra.py "$D"
python3 assemble.py "$D"          # manuscript and supplement as JSON block lists (main.json, supp.json); node build.js converts them to .docx
