# basalt-source-grouped-benchmark

Code for "Source-grouped validation of machine-learning classifiers for basalt tectonic setting".

Data: Appendix 1 of Nakamura (2023), Applied Computing and Geosciences 19, 100132 (`TectonicSetting_1.xlsx`).

Run everything from this folder: `bash run_all.sh /path/to/TectonicSetting_1.xlsx` (several hours on two cores; every step is resumable).
The saved out-of-fold predictions are in `res/oof_predictions.zip`; unzip it into `res/oof/` to rerun `analysis.py` without refitting the models.

| Script | Purpose | Output |
|---|---|---|
| `ml_common.py` | loading, fold-internal ratio selection and scaling, splits, metrics, cluster bootstrap | – |
| `models_ml.py` | the thirteen classifiers and the majority reference, fixed hyperparameters | – |
| `run_baselines.py` | all classifiers, random and source-grouped five-fold | `res/oof/*_44.npy`, `*_24.npy`, `cache/` |
| `leak.py` | shared-code and nearest-neighbour statistics | `res/leak.json` |
| `revision_exp.py` | merged-code partition, alteration-resistant feature set, MgO window, per-code table | `res/revision.json` |
| `rep_grouped.py` | five repetitions with random assignment of codes to folds | `res/repeated_grouped.json` |
| `perm_coherent.py` | permutation importance with ratios recomputed | `res/perm_coherent.csv` |
| `dups.py`, `dedup.py` | near-duplicate analyses; grouped validation without them | `res/dups.json`, `res/dedup.json` |
| `exp_train.py` | augmentation schemes and controls, coarse labels, LightGBM configurations and seeds | `res/exp.jsonl`, `res/oof/x_*.npy` |
| `ternary.py`, `ternary2.py` | nested ternary search for broad groups; arc against other settings | `res/ternary.json`, `res/ternary2.json` |
| `ternary_oceanic.py` | oceanic diagram: nested search, fixed Nb–La–Yb, comparators | `res/ternary_oceanic.json` |
| `analysis.py`, `extra.py` | pooled statistics and bootstrap intervals | `res/V.json`, `res/extra.json` |
| `figs.py`, `fig7.py` | figures | `figs/` |
| `content.py`, `supp.py`, `assemble.py`, `build.js` | assemble manuscript and supplement from the result files | `.docx` |

Software: Python 3.13, NumPy, pandas, SciPy, scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, CatBoost 1.2, Matplotlib, openpyxl.
