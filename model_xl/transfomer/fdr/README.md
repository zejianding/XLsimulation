# FDR analysis (target-decoy competition, 1% line)

Measures how many identifications the **V20 spectral rescoring** produces at a
**1% FDR** cutoff — the "3000 vs 2000" comparison metric.  A better scorer
recovers more spectra as *target* top-1s above the 1% decoy line.

> Natural-language walkthrough of the whole flow: [FDR_METHOD.md](FDR_METHOD.md).
> What every saved CSV/JSON column means: [OUTPUT_COLUMNS.md](OUTPUT_COLUMNS.md).

## Method (TDC — target-decoy competition)

1. **Database.** Every peptide in the species FASTA is expanded into its FULL
   modification variants (fixed C→C+57.021 on every C + variable M→M+15.995 on
   up to 3 M residues) — the same target space the DS database search uses, so
   M-oxidized spectra's true peptides ARE present.  Each variant is a *target*
   (V20 mass table).  Its **reversal** is a *decoy* — reversal preserves the
   exact neutral mass, so a ±ppm precursor window retrieves ~equal numbers of
   targets and decoys, and a decoy can never be a real peptide.  (Options:
   `NO_MOX=1` for plain C+57.021-only fast mode; `VARIANT_FASTA=` a prebuilt DS
   step2 variants FASTA used verbatim.)
2. **Matching.** Each MGF spectrum's precursor mass (PEPMASS/CHARGE →
   neutral, ±10 ppm) retrieves its candidate set — a mix of T and D labeled
   sequences.
3. **Scoring.** The V20 checkpoint predicts each candidate's (39,44,6)
   intensity grid; similarity vs. the real spectrum is measured by:
   - `gc`      grid cosine — (39,44,6) predicted grid vs experimental-projected grid
   - `d`       spectral cosine — *real spectrum projected onto predicted peaks*
   - `f_bidir` spectral cosine — *predicted peaks projected onto real peaks*
     (the bidirectional projection; both directions penalize opposite mistakes)
   - `f_exp` explained intensity, `mby` main b/y precision
   All components are in [0,1], so any branch's fused score = equal-weight raw
   mean of its components is **cross-spectrum comparable** — exactly what the
   FDR ranking needs.  The GPU prediction is done ONCE per candidate and the five
   components are fused per requested branch, so comparing many scoring
   combinations (e.g. `d` vs `gc` vs `d,f_exp` vs `d,gc,f_bidir`) costs no extra
   model forwards.
4. **FDR.** For EACH branch, take every spectrum's top-1 (argmax of that branch's
   fused score), sort descending, accumulate decoy and target hits.
   FDR = decoy/target (Elias–Gygi).  The 1% line = the largest high-scoring
   prefix with q-value ≤ 0.01.  Report `n_target` (identifications), `n_decoy`,
   and `n_target_gt_match` (target top-1s that also mass-match the MGF GT
   sequence) — then compare branches to find the best scoring combination.

## Layout

```
fdr/
  fdr_common.py             shared helpers (decoys, grid predict, scoring, FDR math)
  step1_build_td_fasta.py   species FASTA -> <species>_td.fasta (T + reversed D)
  step2_match_td.py         MGF -> candidates_<species>_td.tsv.gz (labeled)
  step3_score_fdr.py        predict every candidate, score, top-1 per spectrum
  step4_fdr_curve.py        FDR curve, q-values, identification counts at 1%
  run_fdr.sh                one-shot runner (env-parameterized, like ds/run_pipeline.sh)
```

Outputs under `output/<species>/`: `results_step3_*.csv`,
`fdr_curve_*.csv`, `summary_fdr_*.json`, `summary_step3_*.json`, `records/`.

All four steps show `tqdm` progress bars (per-record/spectrum/row with ETA);
`step1` bars cover FASTA reading, target annotation, decoy construction and
FASTA writing; `step2`/`step3` cover the per-spectrum loops; `step4` covers CSV
reading and FDR-curve accumulation.

## Run (server GPU, mouse full)

```bash
cd /home/wangyuhang/seq2mass/v20_0821/transfomer/fdr
bash run_fdr.sh
```

## All species

```bash
SPECIES=all bash run_fdr.sh            # bacillus,clambacteria,honeybee,human,mmazei,
                                       # mouse,ricebean,tomato,yeast  (~22h GPU step3 total)
MAX_SCANS=300 SPECIES=all bash run_fdr.sh    # fast pilot across all species
```

The runner loops over the species list, then aggregates every species' 1% FDR
counts (per preferred branch + each species' best branch) into
`output/summary_fdr_all_species.json` and prints a comparison table.  If a
species has a known precursor bias (ricebean was ~+3.7 ppm), run it separately
with `PPM_OFFSET` — e.g. `PPM_OFFSET=-3.7 SPECIES=ricebean bash run_fdr.sh`.

Pilot on 300 spectra (CPU-friendly):

```bash
MAX_SCANS=300 DEVICE=cpu bash run_fdr.sh
```

Key knobs (env vars, all optional):

| env            | default            | meaning                                    |
|----------------|--------------------|--------------------------------------------|
| `V1_SEARCH_ROOT` | `/home/wangyuhang/seq2mass/v1-search` | data root |
| `CKPT`         | `.../runs/v200701_evidence_v2_massivekb/best_model.pt` | V20 weights |
| `MAX_SCANS`    | *(empty=all)*      | cap spectra per run                        |
| `PPM`          | `10`               | mass window (ppm)                          |
| `NO_MOX`       | `0`                | `1` = plain C+57.021 targets only (fast); `0` = full M-ox variant targets |
| `MAX_VAR_MODS` | `3`                | max M oxidation sites per peptide (target enumeration) |
| `REL`          | `0.10`             | predicted-side intensity filter (try `0.20`) |
| `BRANCHES`     | `d;gc;f_bidir;gc,f_exp;d,gc,f_exp;d,f_exp;d,f_bidir;d,gc,f_bidir` | `;`-separated scoring branches; each is a `,`-subset of gc,d,f_bidir,f_exp,mby |
| `EXP_TRANSFORM`| `sqrt_clip`        | experimental intensity domain (V20 training space) |
| `EXP_TOP_PEAKS`| `0`                | keep top-N experimental peaks before scoring (0 = all) |
| `FDR_TARGETS`  | `0.01,0.05,0.10`   | FDR lines reported                         |
| `SAVE_FULL`    | `1`                | per-candidate full CSV.gz per branch (`full_<branch>_<sp>.csv.gz`) — kept by default so every species' full results are preserved; set `0` to skip (~40-60 MB gz per branch per species) |

Fast iteration without M-ox variants (smaller DB, quicker step3, slightly lower
ceiling — spectra whose true peptide is M-oxidized won't be recoverable):

```bash
NO_MOX=1 bash run_fdr.sh
```

Use a prebuilt DS step2 variants FASTA as the target set verbatim (must exist
on the server):

```bash
VARIANT_FASTA=/home/wangyuhang/seq2mass/v20_0821/transfomer/ds/data/mouse/variants_mouse.fasta bash run_fdr.sh
```

## Reading the output

`summary_fdr_mouse.json` → `branches` → per branch → `fdr_lines."0.01"`:

- `n_target`  — identifications at 1% FDR (the headline count to compare)
- `n_decoy`   — decoys accepted (should be ≈ 1% of n_target)
- `n_target_gt_match` — how many of those are actually the true peptide

Step4 prints a comparison table of all branches sorted by `n_target` at the 1%
line (e.g. `d` alone vs `gc` alone vs `d,f_exp` vs `d,gc,f_bidir`), so you can
pick the scoring combination with the most identifications at the same FDR.
Each branch also gets its own `fdr_curve_<species>_<branch>.csv`.

For post-hoc score analysis step3 writes one per-candidate CSV.gz PER BRANCH by
default (`SAVE_FULL=1`) — `full_<branch>_<species>.csv.gz` — with, for
EVERY candidate of every spectrum: scan info (ordinal/scan_id/charge/pepmass/rt/
gt_seq), the candidate itself (cand_idx/seq/label/ppm/neutral_mass), all five
component scores (gc/d/f_bidir/f_exp/mby), the branch fused score, the
within-spectrum rank, `is_best` (rank 1) and `gt_match`.  This is where you can
look at, e.g., the actual gc score distribution of targets vs decoys, or how
often the GT sits at rank 1 vs rank 2, without re-running predictions.

**Recompute any branch from a saved full CSV (no GPU).** Since every full CSV
carries all five component scores per candidate, step4 can re-derive the FDR for
ANY new component combination in seconds — e.g. to answer "what about gc+f_exp?"
without a 2h GPU re-run:

```bash
python step4_fdr_curve.py --species mouse --out-dir output/mouse \
    --full output/mouse/full_d_gc_f_bidir_mouse.csv.gz \
    --branches 'gc,f_exp;d,gc,f_exp;d,f_exp;gc;d'
```

New branches share the rel/exp-transform that the saved CSV was scored with.

Compare two scorers on the same species/MGF/FASTA: the one with the larger
`n_target` at 1% FDR wins.  FDR is convention-independent for the comparison,
as long as both use the same decoy set and FDR formula.
