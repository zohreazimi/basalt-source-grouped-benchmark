import json, os, numpy as np, pandas as pd
from collections import Counter

def build(C, df):
    V, Mo, CAL, AUG, H, RV, TO = C.V, C.Mo, C.CAL, C.AUG, C.H, C.RV, C.TO
    NICE = C.NICE
    SECS = {}; cur = ['head']
    def A(blk): SECS.setdefault(cur[0], []).append(blk)
    def SEC(k): cur[0] = k
    P = lambda t: A({'type': 'p', 'text': t})
    A({'type': 'title', 'text': 'Supplementary material'})
    P('**' + C.TITLE + '**')
    P('Zohre Azimi, Jamshid Saeidian, Hossein Azimi, Mohsen Azimi')
    SEC('S2')
    A({'type': 'caption', 'text': '**Table S2.** Procedures, classifiers and hyperparameters. Settings not listed are library defaults (scikit-learn 1.9, LightGBM 4.7, XGBoost 3.4, CatBoost 1.2). The random seed was 42 unless stated. Hyperparameters were fixed in advance and not tuned.', 'keepNext': True})
    hp = [['Item', 'Settings'],
     ['Ratio selection', '276 pairwise log-ratios; scores: ANOVA F, mutual information (3 nearest neighbours), impurity importance of a 300-tree random forest with balanced class weights; each min–max normalised and averaged; ratios accepted in rank order unless |Pearson r| > 0.98 with an accepted ratio; 20 retained; fitted on the outer training fold'],
     ['Transformation', 'Yeo–Johnson power transformation followed by standardisation, fitted on the outer training fold'],
     ['Splits', 'stratified five-fold (random protocol) and stratified group five-fold with the reference code as group (grouped protocol), both shuffled with seed 42'],
     ['LightGBM', '600 trees, learning rate 0.05, 31 leaves, row subsampling 0.8 (every iteration), column subsampling 0.8, balanced class weights'],
     ['LightGBM, sensitivity runs', 'small: 300 trees, 7 leaves; medium: 300 trees, 15 leaves; large: 1,000 trees, 63 leaves; slow: 1,500 trees, 31 leaves, learning rate 0.02; otherwise as above'],
     ['XGBoost', '600 trees, learning rate 0.05, maximum depth 6, row subsampling 0.8, column subsampling 0.8, histogram tree method; no class weights'],
     ['CatBoost', '800 iterations, learning rate 0.08, depth 6, balanced class weights'],
     ['Random forest', '500 trees, balanced class weights'],
     ['Extremely randomised trees', '500 trees, balanced class weights'],
     ['k-nearest neighbours', 'k = 5, inverse-distance weighting, Euclidean distance; no class weights'],
     ['SVM (RBF kernel)', 'C = 10, γ = 1/(number of features × feature variance), balanced class weights, probabilities by Platt scaling (internal five-fold, not grouped); predicted class = most probable class'],
     ['SVM (linear kernel)', 'C = 1, balanced class weights, probabilities by 3-fold sigmoid calibration'],
     ['Logistic regression', 'C = 1, multinomial, balanced class weights'],
     ['Linear discriminant analysis', 'defaults (empirical priors)'],
     ['Decision tree', 'unpruned, balanced class weights'],
     ['AdaBoost', '300 estimators; no class weights'],
     ['Gaussian naive Bayes', 'defaults'],
     ['Majority-class predictor', 'most frequent class of the training fold'],
     ['Soft-voting ensemble', 'mean of the class probabilities of LightGBM, SVM (RBF kernel) and CatBoost'],
     ['Augmentation', 'training fold only; classes with fewer than 1,500 training samples filled to 1,500 (also 750 and 3,000 for cross-source mixing); seeds 1, 2, 3; synthetic analyses generated in log-concentration space, ratios recomputed, oxides not re-closed, features transformed with the transformer fitted on the real training samples; duplication: copies of randomly drawn analyses; noise: log-normal, σ = 0.02 (major oxides) and 0.08 (trace elements); interpolation weight λ ~ U(0, 1); nearest-neighbour partner among the 5 nearest same-class neighbours by Euclidean distance in unscaled log-concentration space; within-source partner drawn at random from same-class samples with the same reference code (possibly the sample itself); cross-source partner drawn at random from same-class samples with a different reference code; “noise, all classes”: one noisy copy of every training sample'],
     ['Near-duplicates', 'pairs of analyses in which each of the 16 trace elements differs by no more than 1% (ratio within 1.01)'],
     ['Bootstrap', '10,000 resamples of reference codes with replacement, seed 0, the same resamples for all models; percentile intervals'],
     ['Ternary search', '364 triples of TiO₂, Zr, Hf, Nb, Ta, Th, Y, La, Ce, Nd, Sm, Gd, Yb, Lu; linear discriminant analysis (least-squares solver) with equal priors on the two log-ratios; inner selection by three-fold source-grouped cross-validation (seed 0) inside each outer source-grouped fold (seed 42)']]
    A({'type': 'table', 'rows': hp, 'widths': [0.26, 0.74], 'left': True, 'size': 17})

    SEC('S4')
    A({'type': 'caption', 'text': '**Table S4.** All models under both protocols (fold mean ± standard deviation over five folds). Acc.: accuracy (%); BAcc.: balanced accuracy (%); F1: macro-averaged F1 (%); MCC: Matthews correlation coefficient; AUC: macro-averaged one-versus-rest area under the ROC curve. “[24]” denotes models trained on the 24 concentrations without ratios.', 'keepNext': True})
    rows = [['Model', 'Protocol', 'Acc.', 'BAcc.', 'F1', 'MCC', 'AUC']]
    for m in sorted(Mo, key=lambda m: -Mo[m]['grouped']['bacc'][0]):
        nm = NICE.get(m.replace(' [24]', ''), m.replace(' [24]', '')) + (' [24]' if '[24]' in m else '')
        for prot, pl in [('random', 'Random'), ('grouped', 'Grouped')]:
            v = Mo[m][prot]; rows.append([nm, pl, f"{100*v['acc'][0]:.1f} ± {100*v['acc'][1]:.1f}", f"{100*v['bacc'][0]:.1f} ± {100*v['bacc'][1]:.1f}", f"{100*v['f1_macro'][0]:.1f} ± {100*v['f1_macro'][1]:.1f}", f"{v['mcc'][0]:.3f} ± {v['mcc'][1]:.3f}", f"{v['auc_m'][0]:.3f} ± {v['auc_m'][1]:.3f}"])
    A({'type': 'table', 'rows': rows, 'widths': [0.24, 0.09, 0.13, 0.13, 0.13, 0.14, 0.14], 'size': 15})

    cnt = {}
    for prot in ['random', 'grouped']:
        c = Counter()
        for f in range(5): c.update(list(np.load(f'cache/{prot}_{f}.npz', allow_pickle=True)['feat'])[24:])
        cnt[prot] = c
    allr = sorted(set(cnt['random']) | set(cnt['grouped']), key=lambda r: -(cnt['random'][r] + cnt['grouped'][r]))
    SEC('S3')
    A({'type': 'caption', 'text': '**Table S3.** Log-ratios selected by the consensus ranking: number of folds (out of five) in which each ratio was among the 20 retained. Ratios are written in the fixed element order of the dataset; because log-ratios are used, the direction is immaterial.', 'keepNext': True})
    rows = [['Ratio', 'Random folds', 'Grouped folds', 'Ratio', 'Random folds', 'Grouped folds']]; half = (len(allr) + 1) // 2
    for i in range(half):
        l = allr[i]; r = allr[i + half] if i + half < len(allr) else None
        rows.append([C.chem(l), str(cnt['random'][l]), str(cnt['grouped'][l])] + ([C.chem(r), str(cnt['random'][r]), str(cnt['grouped'][r])] if r else ['', '', '']))
    A({'type': 'table', 'rows': rows, 'widths': [0.2, 0.15, 0.15, 0.2, 0.15, 0.15], 'size': 17})

    CIx = V['ci']
    SEC('DROP')
    A({'type': 'caption', 'text': '**Table DROP.** Pooled out-of-fold estimates with 95% bootstrap intervals (reference codes resampled): accuracy and balanced accuracy under source-grouped validation, and the loss relative to random splits (percentage points).', 'keepNext': True})
    rows = [['Model', 'Grouped accuracy (%)', 'Grouped balanced acc. (%)', 'Loss of accuracy', 'Loss of balanced acc.']]
    for m in ['LightGBM', 'CatBoost', 'SVM (RBF)', 'XGBoost', 'Random Forest', 'KNN', 'LDA', 'Ensemble']:
        v = CIx[m]; rows.append([NICE.get(m, m), C.f3(v['acc']), C.f3(v['bacc']), C.f3(v['gap_acc']), C.f3(v['gap_bacc'])])
    A({'type': 'table', 'rows': rows, 'widths': [0.28, 0.18, 0.18, 0.18, 0.18], 'size': 17})

    SEC('S8')
    A({'type': 'caption', 'text': '**Table S8.** Calibration and selective prediction. ECE: expected calibration error; Conf.: mean confidence; NLL: negative log-likelihood, each under grouped (random) validation. Coverage: share of samples with maximum class probability at or above the threshold under grouped validation, with accuracy and balanced accuracy of those samples (%).', 'keepNext': True})
    rows = [['Model', 'ECE; Conf.; NLL', 'Threshold', 'Coverage', 'Accuracy', 'Balanced accuracy']]
    for k in ['SVM (RBF)', 'CatBoost', 'LightGBM', 'XGBoost', 'Random Forest', 'Ensemble']:
        cc = CAL[k]; info = {'0.5': f"{cc['grouped']['ece']:.3f} ({cc['random']['ece']:.3f})", '0.7': f"{cc['grouped']['conf']:.3f} ({cc['random']['conf']:.3f})", '0.9': f"{cc['grouped']['nll']:.2f} ({cc['random']['nll']:.2f})"}
        for th in ['0.5', '0.7', '0.9', '0.95']:
            s = cc['sel'][th]; rows.append([NICE.get(k, k) if th == '0.5' else '', info.get(th, ''), th, f"{100*s['coverage']:.1f}", f"{100*s['acc']:.1f}", f"{100*s['bacc']:.1f}"])
    A({'type': 'table', 'rows': rows, 'widths': [0.26, 0.18, 0.13, 0.13, 0.13, 0.17], 'size': 17})

    pc = C.PC.groupby(['kind', 'item'])['drop'].agg(['mean', 'std'])
    SEC('S9')
    A({'type': 'caption', 'text': '**Table S9.** Permutation importance for LightGBM on held-out source-grouped folds: loss of balanced accuracy (percentage points, mean ± standard deviation over five folds) when the raw concentrations of an element or block are shuffled among test samples and all ratios are recomputed.', 'keepNext': True})
    f = lambda k, it: (lambda mu, sd: f"{0.0 if abs(mu) < 0.05 else mu:.1f} ± {sd:.1f}")(100*pc.loc[(k, it), 'mean'], 100*pc.loc[(k, it), 'std'])
    el = list(pc.loc['element'].sort_values('mean', ascending=False).index); bl = ['all 24 concentrations', '16 trace elements', '8 major oxides', 'HFSE (TiO2, Zr, Hf, Nb, Ta)', 'LILE (K2O, Rb, Ba, Sr)', 'REE and Y']
    items = [(C.chem(it[0].upper() + it[1:]), f('block', it)) for it in bl] + [(C.chem(it), f('element', it)) for it in el]; half = (len(items) + 1) // 2
    rows = [['Element or block', 'Loss', 'Element or block', 'Loss']] + [[items[i][0], items[i][1]] + (list(items[i + half]) if i + half < len(items) else ['', '']) for i in range(half)]
    A({'type': 'table', 'rows': rows, 'widths': [0.32, 0.18, 0.32, 0.18], 'size': 17})

    md = RV['medians']
    SEC('S1')
    A({'type': 'caption', 'text': '**Table S1.** Class medians of selected ratios and oxides (wt%) in the dataset.', 'keepNext': True})
    rows = [['Setting', 'Nb/La', 'Th/Nb', 'Ba/Nb', 'TiO₂', 'MgO', 'Fe₂O₃t', 'K₂O']]
    for c in ['MORB', 'OPB', 'OIB', 'CFB', 'CAB', 'BABB', 'IAB', 'IOAB']:
        v = md[c]; rows.append([c, f"{v['NbLa']:.2f}", f"{v['ThNb']:.2f}", f"{v['BaNb']:.0f}", f"{v['TiO2']:.2f}", f"{v['MgO']:.1f}", f"{v['Fe2O3t']:.1f}", f"{v['K2O']:.2f}"])
    A({'type': 'table', 'rows': rows, 'widths': [0.16, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12], 'size': 17})

    RG = C.RG
    SEC('S5')
    A({'type': 'caption', 'text': '**Table S5.** Sensitivity of the grouped estimate (%). “Main partition”: stratified group five-fold on reference codes. “Random assignment”: codes assigned to five folds at random, five repetitions (range of pooled values). “Merged codes”: codes sharing a constituent reference merged into connected components before splitting. “Alteration-resistant elements”: TiO₂, Zr, Hf, Nb, Ta, Th, Y and rare earth elements with all pairwise log-ratios. “MgO 5–10 wt%”: training and testing restricted to that window. “Near-duplicates removed”: one member of each near-duplicate pair and both members of pairs with conflicting labels removed, duplicate-linked codes merged, new grouped partition. “Other seeds”: LightGBM refitted with random seeds 1–3 (range). LightGBM configurations are defined in Table S2. All values are pooled over folds, as in Table 2, except the MgO row.', 'keepNext': True})
    rows = [['Experiment', 'Model', 'Accuracy', 'Balanced acc.', 'MORB', 'OIB', 'BABB', 'OPB', 'IOAB']]
    for m in ['LightGBM', 'SVM (RBF)', 'CatBoost']:
        pcl = V['per_class'][f'grouped|{m}']['recall']; r = lambda c: f"{100*pcl[C.classes.index(c)]:.0f}"
        rows.append(['Main partition', NICE.get(m, m), f"{100*V['ci'][m]['acc'][0]:.1f}", f"{100*V['ci'][m]['bacc'][0]:.1f}", r('MORB'), r('OIB'), r('BABB'), r('OPB'), r('IOAB')])
    sv = V['seedvar']; rows.append(['Other seeds (3 refits)', 'LightGBM', f"{100*min(sv['acc'][:3]):.1f}–{100*max(sv['acc'][:3]):.1f}", f"{100*min(sv['bacc'][:3]):.1f}–{100*max(sv['bacc'][:3]):.1f}", '–', '–', '–', '–', '–'])
    DD = V['dups'].get('dedup', {})
    for m in ['LightGBM', 'SVM (RBF)', 'CatBoost']:
        if m in DD: v = DD[m]; rows.append([f"Near-duplicates removed (n = {DD['n_keep']:,})", NICE.get(m, m), C.f3(v['acc_ci']), C.f3(v['bacc_ci'])] + [f"{100*v['recall'][c]:.0f}" for c in ['MORB', 'OIB', 'BABB', 'OPB', 'IOAB']])
    for m in ['LightGBM', 'SVM (RBF)']:
        g = RG[RG.model == m]; rr = lambda c, d=0: f"{100*g[c].min():.{d}f}–{100*g[c].max():.{d}f}"
        rows.append(['Random assignment (5 repetitions)', NICE.get(m, m), rr('acc', 1), rr('bacc', 1), rr('MORB'), rr('OIB'), rr('BABB'), rr('OPB'), rr('IOAB')])
    def rvrow(lab, key, models):
        for m in models:
            v = RV[key][m]; rc = v['recall']; pv_ = V['rev_pooled'][{'component_grouped': 'comp'}.get(key, key) + '|' + m]; rows.append([lab, NICE.get(m, m), f"{100*pv_['acc']:.1f}", f"{100*pv_['bacc']:.1f}"] + [f"{100*rc[c]:.0f}" for c in ['MORB', 'OIB', 'BABB', 'OPB', 'IOAB']])
    rvrow('Merged codes', 'component_grouped', ['LightGBM', 'SVM (RBF)', 'CatBoost'])
    rvrow('Alteration-resistant elements, random splits', 'immobile_random', ['LightGBM', 'SVM (RBF)'])
    rvrow('Alteration-resistant elements, main grouped partition', 'immobile_grouped', ['LightGBM', 'SVM (RBF)'])
    rvrow('Alteration-resistant elements, merged codes', 'immobile_comp', ['LightGBM', 'SVM (RBF)'])
    rows.append(['MgO 5–10 wt%, main grouped partition (fold means)', 'LightGBM', f"{100*RV['mgo']['LightGBM']['acc'][0]:.1f}", f"{100*RV['mgo']['LightGBM']['bacc'][0]:.1f}", '–', '–', '–', '–', '–'])
    for cfg in ['small', 'medium', 'large', 'slow']:
        for p, pl in [('random', 'random splits'), ('grouped', 'main grouped partition')]:
            v = V['lgb_pooled'].get(f'{p}|{cfg}')
            if v: rows.append([f'LightGBM “{cfg}”, {pl}', 'LightGBM', f"{100*v['acc']:.1f}", f"{100*v['bacc']:.1f}", '–', '–', '–', '–', '–'])
    A({'type': 'table', 'rows': rows, 'widths': [0.26, 0.12, 0.135, 0.135, 0.07, 0.07, 0.07, 0.07, 0.07], 'size': 15})

    SEC('S7')
    A({'type': 'caption', 'text': '**Table S7.** LightGBM under source-grouped validation for individual reference codes with at least 15 analyses of OPB, BABB or IOAB: number of analyses of that setting, recall, and the most frequent wrong class.', 'keepNext': True})
    rows = [['Setting', 'Reference code', 'Database', 'n', 'Recall (%)', 'Main wrong class']]
    for r in RV['per_code']: rows.append([r['cls'], r['code'] + (' †' if r['code'] in ('[19104]', 'SANO, 2012') else ' ‡' if any(k in r['code'] for k in ('TAMURA, 2011', 'HICKEY-VARGAS, 2018', 'ISHIZUKA, 2009')) else ''), 'GEOROC' if r['db'] == 'GeoROC' else 'PetDB', str(r['n']), f"{100*r['recall']:.0f}", r['main_wrong']])
    A({'type': 'table', 'rows': rows, 'widths': [0.12, 0.34, 0.14, 0.10, 0.14, 0.16], 'size': 17})
    A({'type': 'note', 'text': '† The two codes list 79 of the same drill-core samples (near-duplicates); each is in the training data when the other is tested. ‡ Codes whose back-arc affinity should be checked against the source papers; TAMURA, 2011 shares samples with GEOROC code [15516], labelled IOAB.'})

    SEC('S10')
    A({'type': 'caption', 'text': '**Table S10.** Label hierarchy under source-grouped validation (pooled out-of-fold predictions, %; 95% bootstrap intervals): the eight class probabilities are summed within groups. Six groups: the three arc settings merged. Four groups: arc, ridge (MORB, BABB), oceanic intraplate (OIB, OPB), CFB. Three groups: arc, ridge, intraplate (OIB, OPB, CFB). κ: Cohen’s kappa. Loss: random-split minus grouped balanced accuracy (points). “Direct”: LightGBM trained on the coarse labels.', 'keepNext': True})
    rows = [['Model', 'Groups', 'Accuracy', 'Balanced accuracy', 'κ', 'Loss']]
    for m, lab in [('LightGBM', 'LightGBM'), ('Ensemble', 'Soft-voting ensemble')]:
        for s_ in ['8', '6', '4', '3']:
            h = H[f'grouped|{m}|{s_}']; rows.append([lab if s_ == '8' else '', s_, C.f3(h['acc']), C.f3(h['bacc']), f"{h['kappa']:.3f}", C.f3(h['gap_bacc'])])
    for k in ['4', '3']: rows.append(['LightGBM, direct' if k == '4' else '', k, f"{100*H[f'grouped|direct|{k}']['acc']:.1f}", f"{100*H[f'grouped|direct|{k}']['bacc']:.1f}", '–', f"{100*(H[f'random|direct|{k}']['bacc']-H[f'grouped|direct|{k}']['bacc']):.1f}"])
    A({'type': 'table', 'rows': rows, 'widths': [0.24, 0.08, 0.19, 0.19, 0.10, 0.20], 'size': 16, 'noteAfter': 60})
    rows = [['Share of analyses assigned to the correct group (LightGBM, %)'] + C.classes]
    for s_, nm in [('6', 'Six groups'), ('4', 'Four groups'), ('3', 'Three groups')]: rows.append([nm] + [f"{100*H[f'grouped|LightGBM|{s_}']['fine_to_group'][c]:.0f}" for c in C.classes])
    A({'type': 'table', 'rows': rows, 'widths': [0.36] + [0.08] * 8, 'size': 16})
    r3, r4 = H['randmerge|3'], H['randmerge|4']
    P(f"Reference for the merging effect: among all {r3['n']} ways of merging the eight settings into groups of sizes 3, 3 and 2, κ of the summed LightGBM probabilities has a median of {r3['kappa_median']:.3f} and a maximum of {r3['kappa_max']:.3f}, and the geological three-group scheme ranks {r3['kappa_rank']}; among the {r4['n']} groupings of sizes 3, 2, 2 and 1 the median is {r4['kappa_median']:.3f}, the maximum {r4['kappa_max']:.3f}, and the geological four-group scheme ranks {r4['kappa_rank']}.")

    SEC('S11')
    A({'type': 'h2', 'text': 'Note S1. Augmentation schemes'})
    P(open('aug_note.txt').read())
    A({'type': 'caption', 'text': '**Table S11.** Augmentation of minority classes: all variants run (LightGBM unless stated; pooled out-of-fold predictions, mean of three seeds unless stated): source-grouped validation, random splits, and paired differences in balanced accuracy between schemes. Changes are relative to the unaugmented model, in percentage points, with 95% paired bootstrap intervals.', 'keepNext': True})
    rows = [['Variant', 'Accuracy (%)', 'Balanced acc. (%)', 'Δ accuracy', 'Δ balanced acc.']]
    lab = {'xsource': 'Cross-source mixing, target 1,500', 'xsource|750': 'Cross-source mixing, target 750', 'xsource|3000': 'Cross-source mixing, target 3,000', 'within': 'Within-source mixing', 'noise': 'Analytical noise, minority classes', 'noise_all': 'Analytical noise, one copy of every sample', 'smote': 'Nearest-neighbour interpolation', 'dup': 'Duplication (control)'}
    for mn, ml in [('LightGBM', ''), ('SVM (RBF)', 'SVM (RBF kernel): '), ('CatBoost', 'CatBoost: ')]:
        for k, nm in lab.items():
            a = AUG.get(f'grouped|{mn}|{k}')
            if a: rows.append([ml + (nm if not ml else nm[0].lower() + nm[1:]) + ('' if a['n_seeds'] == 3 else f" ({a['n_seeds']} seed)"), f"{100*a['acc']:.1f}", f"{100*a['bacc']:.1f}", C.f3(a['d_acc'], sign=True), C.f3(a['d_bacc'], sign=True)])
    A({'type': 'table', 'rows': rows, 'widths': [0.40, 0.13, 0.13, 0.17, 0.17], 'size': 16, 'noteAfter': 60})
    rows = [['Random splits (LightGBM unless stated)', 'Δ accuracy', 'Δ balanced acc.']]
    for mn, ml in [('LightGBM', ''), ('SVM (RBF)', 'SVM (RBF kernel): '), ('CatBoost', 'CatBoost: ')]:
        for k, nm in lab.items():
            a = AUG.get(f'random|{mn}|{k}')
            if a: rows.append([ml + (nm if not ml else nm[0].lower() + nm[1:]) + ('' if a['n_seeds'] == 3 else f" ({a['n_seeds']} seed)"), C.f3(a['d_acc'], sign=True), C.f3(a['d_bacc'], sign=True)])
    A({'type': 'table', 'rows': rows, 'widths': [0.5, 0.25, 0.25], 'size': 16, 'noteAfter': 60, 'left': True})
    cn = {'xsource|noise': 'Cross-source mixing − analytical noise', 'xsource|smote': 'Cross-source mixing − nearest-neighbour interpolation', 'xsource|within': 'Cross-source mixing − within-source mixing', 'xsource|dup': 'Cross-source mixing − duplication', 'noise|dup': 'Analytical noise − duplication', 'noise|smote': 'Analytical noise − nearest-neighbour interpolation', 'xsource LightGBM|plain CatBoost': 'LightGBM with cross-source mixing − CatBoost without augmentation'}
    rows = [['Paired difference (LightGBM, grouped)', 'Balanced accuracy (points)']] + [[nm, C.f3(AUG['contrasts'][k], sign=True)] for k, nm in cn.items() if k in AUG['contrasts']]
    A({'type': 'table', 'rows': rows, 'widths': [0.65, 0.35], 'size': 16, 'left': True})

    SEC('S6')
    lk = V['linked']['by_class']
    A({'type': 'caption', 'text': '**Table S6.** Recall of LightGBM under source-grouped validation (main partition, %) for test samples whose reference code shares a constituent reference with a code in the training fold (“linked”) and for the remaining test samples.', 'keepNext': True})
    rows = [['Setting', 'Linked: n', 'Linked: recall', 'Not linked: n', 'Not linked: recall']]
    for c in C.classes:
        v = lk[c]; rows.append([c, f"{v['n_linked']:,}", f"{100*v['rec_linked']:.0f}" if v['rec_linked'] is not None else '–', f"{v['n_unlinked']:,}", f"{100*v['rec_unlinked']:.0f}"])
    A({'type': 'table', 'rows': rows, 'widths': [0.2, 0.2, 0.2, 0.2, 0.2], 'size': 17})

    SEC('S12')
    co, ic = TO['coef'], TO['intercept']; SF = TO['same_folds']; tf = SF['Nb–La–Yb (triple fixed)']
    A({'type': 'caption', 'text': f"**Table S12.** The Nb–La–Yb diagram for oceanic basalts. (a) Discriminant scores *s* = *c*₀ + *c*₁ ln(Nb/La) + *c*₂ ln(La/Yb), with concentrations in ppm; a sample is assigned to the class with the largest score. The functions were fitted by linear discriminant analysis with equal priors to all {TO['n']:,} oceanic-arc (IAB, IOAB), MORB and OIB analyses; recall and precision are from source-grouped five-fold cross-validation with the triple fixed. (b) Share of analyses falling in each field (%), for the settings not used in the fit (all analyses) and for subgroups of the classes used (source-grouped predictions). (c) Nested search for three definitions of the classes. (d) Coordinates and models evaluated on the outer folds of the first definition, all with source-grouped validation unless stated; the last column is the paired difference in accuracy from the nested search (points). Intervals are 95% bootstrap intervals (reference codes resampled).", 'keepNext': True})
    rows = [['(a) Class', 'c₀', 'c₁ (ln Nb/La)', 'c₂ (ln La/Yb)', 'n', 'Recall (%)', 'Precision (%)']]
    for k, nm in enumerate(['Oceanic arc', 'MORB', 'OIB + enriched MORB']):
        rows.append([nm, C.mn_(f"{ic[k]:.3f}"), C.mn_(f"{co[k][0]:.3f}"), C.mn_(f"{co[k][1]:.3f}"), f"{TO['n_by_class'][k]:,}", f"{100*tf['recall'][k]:.1f}", f"{100*tf['precision'][k]:.1f}"])
    A({'type': 'table', 'rows': rows, 'widths': [0.22, 0.11, 0.15, 0.15, 0.11, 0.13, 0.13], 'size': 16, 'noteAfter': 60})
    rows = [['(b) Setting or subgroup', 'Oceanic-arc field', 'MORB field', 'OIB field']]
    for c in ['BABB', 'OPB', 'CFB', 'CAB']: rows.append([c + ' (not used in the fit)'] + [f"{100*x:.0f}" for x in TO['excluded'][c]])
    sb = TO['subgroups']
    for k in ['IAB', 'IOAB', 'MORB, (La/Sm)N < 1', 'MORB, (La/Sm)N ≥ 1', 'OIB']: rows.append([f"{k} (used in the fit; n = {sb[k]['n']:,})"] + [f"{100*x:.0f}" for x in sb[k]['fields_fixed']])
    for k, nm in [('1-1.5', '1 ≤ (La/Sm)N < 1.5'), ('1.5-2', '1.5 ≤ (La/Sm)N < 2'), ('>=2', '(La/Sm)N ≥ 2')]:
        e = C.EX['morb_bins'][k]; rows.append([f"MORB, {nm} (final fields; n = {e['n']:,})", '', f"{100*e['morb_field']:.0f}", ''])
    A({'type': 'table', 'rows': rows, 'widths': [0.43, 0.19, 0.19, 0.19], 'size': 16, 'noteAfter': 60})
    rows = [['(c) Classes in the nested search', 'n', 'Accuracy (%)', 'Balanced accuracy (%)', 'Triples chosen in the five outer folds']]
    for k in [k for k in TO if k.startswith('nested|')]:
        v = TO[k]; rows.append([k.split('|')[1], f"{v['n']:,}", C.f3(v['acc']), C.f3(v['bacc']), '; '.join(v['chosen'])])
    A({'type': 'table', 'rows': rows, 'widths': [0.25, 0.08, 0.16, 0.16, 0.35], 'size': 15, 'noteAfter': 60})
    refs = {'Nested selection': 'Nested selection among 364 triples', 'Nb–La–Yb (triple fixed)': 'Nb–La–Yb, triple fixed (not nested)', 'Nb–La–Yb, random splits': 'Nb–La–Yb, triple fixed, random stratified five-fold', 'QDA, Nb–La–Yb': 'Nb–La–Yb, quadratic discriminant analysis', 'La–Y–Nb': 'La–Y–Nb (Cabanis and Lecolle, 1989)', 'Th–Nb–Yb': 'Th–Nb–Yb (elements of Pearce, 2008)', 'Th–Hf–Ta': 'Th–Hf–Ta (Wood, 1980)', 'Nb–Zr–Y': 'Nb–Zr–Y (Meschede, 1986)', 'Ti–Zr–Y': 'Ti–Zr–Y (Pearce and Cann, 1973)', 'LDA, 14 immobile elements': 'Linear discriminant analysis, 14 log-concentrations', 'LightGBM, 14 immobile elements': 'LightGBM, 14 alteration-resistant elements', 'LightGBM, 24 concentrations': 'LightGBM, 24 concentrations'}
    rows = [['(d) Coordinates or model', 'Accuracy (%)', 'Balanced accuracy (%)', 'Accuracy minus nested (points)']]
    for k, nm in refs.items():
        v = SF[k]; rows.append([nm, C.f3(v['acc']), C.f3(v['bacc']), '–' if k == 'Nested selection' else C.f3(v['minus_nested']['acc'], sign=True)])
    A({'type': 'table', 'rows': rows, 'widths': [0.40, 0.20, 0.20, 0.20], 'size': 16, 'left': True})
    P("Values for the published diagrams describe the information carried by their coordinates under the same linear classifier, not the published field boundaries, and those diagrams were designed for other sets of classes. (La/Sm)N is normalised to the chondrite of McDonough and Sun (1995). The three field boundaries meet at Nb/La = " + f"{C.EX['triple']['NbLa']:.2f} and La/Yb = {C.EX['triple']['LaYb']:.1f}." + " The results depend slightly on the numerical solver of the discriminant analysis: with the singular-value-decomposition solver of scikit-learn the fixed Nb–La–Yb diagram reaches " + f"{100*C.EX['fixed_svd_acc']:.1f}% accuracy, against {100*tf['acc'][0]:.1f}% with the least-squares solver used throughout.")

    SEC('figs')
    A({'type': 'figure', 'path': f'{C.FIG}/figS1.png', 'w': C.FS['figS1'][0], 'h': C.FS['figS1'][1], 'width': 600})
    A({'type': 'caption', 'text': f"**Fig. S1.** Exploratory views of the dataset. (a) First two principal components of the 24 Yeo–Johnson-transformed, standardised concentrations (all {V['data']['n']:,} analyses), coloured by tectonic setting. (b) Skewness of the raw concentrations; the dashed line marks a skewness of 1."})
    A({'type': 'figure', 'path': f'{C.FIG}/figS2.png', 'w': C.FS['figS2'][0], 'h': C.FS['figS2'][1], 'width': 600})
    A({'type': 'caption', 'text': '**Fig. S2.** Row-normalised confusion matrices (%) under source-grouped validation for (a) the RBF support vector machine and (b) CatBoost, from pooled out-of-fold predictions.'})
    A({'type': 'figure', 'path': f'{C.FIG}/figS3.png', 'w': C.FS['figS3'][0], 'h': C.FS['figS3'][1], 'width': 560})
    A({'type': 'caption', 'text': '**Fig. S3.** Settings that were not used to define the Nb–La–Yb diagram, plotted in its coordinates with the field boundaries of Fig. 7b. The panel titles give the share of analyses falling in each field.'})
    b = []
    for k in ['head'] + [f'S{i}' for i in range(1, 13)] + ['figs']: b += SECS[k]
    json.dump(b, open('supp.json', 'w'), ensure_ascii=False)
