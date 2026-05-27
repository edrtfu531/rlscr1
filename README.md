This repo is organized around **experiment universes** (run folders). In all commands below, replace:

- `run_00_market_proxy` with the run folder you want, e.g.
  - `run_00_market_proxy`
  - `run_00_high_vol` … `run_09_high_vol`
  - `run_10_low_vol` … `run_19_low_vol`
  - `run_20_general` … `run_29_general`

In the examples, we use `run_00_market_proxy`.
The equities in each universe are selected using `code/data/market/splits.ipynb`.

The release focuses on portfolio optimization, SCR and critic-augmentation experimental artifacts.

---

## Requirements

- **Python:** tested with Python **3.9.13**
- **Packages:** install from `requirements.txt` (includes core ML deps like `torch`, `torch-geometric`, and `transformers`)

```bash
pip install -r requirements.txt
```

---

## 1) Data Sources

- S&P 500 index data are obtained from Yahoo Finance using ticker `^GSPC`.

- Historical S&P 500 constituent data are obtained from the public GitHub repository `fja05680/sp500`: https://github.com/fja05680/sp500.

- Market price data are obtained from FNSPID on Hugging Face (`Zihan1004/FNSPID`).

- Macro-financial series are obtained from FRED, Federal Reserve Bank of St. Louis. These include high-yield spreads, Baa–10Y spreads, Treasury-based shock features, the 2s10s Treasury spread, WTI, the U.S. Dollar Index, VIX, U.S. Economic Policy Uncertainty, Brent, energy. 

  All direct macro-financial series are processed using date parsing, numeric cleaning, missing-value handling, and alignment to the experiment calendar. Treasury-based variables are derived separately:

  | Category | Variable | Type | Source | Preprocessing / transformation |
  |---|---|---|---|---|
  | Treasury yield curve | 10Y–2Y Treasury spread | Derived | FRED Treasury yields | Computed as the 10-year Treasury yield minus the 2-year Treasury yield, then aligned to the experiment calendar |
  | Treasury shock | 10-year Treasury shock feature | Derived | FRED Treasury yields | Computed from daily changes in the 10-year Treasury yield and standardized using a rolling historical window |

- FOMC statements are obtained from the Federal Reserve official website: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm.

- The Geopolitical Risk Index is obtained from the Economic Policy Uncertainty website: https://www.policyuncertainty.com/gpr.html.

Original third-party datasets are not redistributed in this repository. Users must obtain any required third-party datasets from the relevant original providers and follow their applicable terms and licenses.

From `code/`, run this to generate return panel:

```bash
python generate_returns.py \
  --config code/output/configs/config_returns.yaml \
  --universe_json data/market/market_proxy.json \
  --run_id 0
```

To run a specific universe, change `--run_id` accordingly. Use `market_proxy.json` for the market-proxy experiment, and use `experiment_universes_selected_runs.json` for the other universes.  Next steps can be run once the relevant data sources and preprocessing outputs are available.

---

## 2) Scenario-Context Rollout (SCR) build steps

From `code/scr/`, run the following **in order**.

### 2.1 Build Fed signal (FinBERT embeddings)

```bash
python build_fedsignal.py \
  --emb_model "yiyanghkust/finbert-pretrain" \
  --fomc_dir ../data/fomc_statements \
  --out ../data/fomc_statements/fedsignal.csv \
  --returns_csv ../output/experiments/run_00_market_proxy/returns_run_00_market_proxy.csv
```

### 2.2 Build macro shock indices

```bash
python build_shock_indices.py \
  --returns_csv ../output/experiments/run_00_market_proxy/returns_run_00_market_proxy.csv \
  --macro_dir   ../data/macro \
  --out_csv     ../data/macro/shock_indices.csv
```

### 2.3 Tier 1: unknown-shock discovery

```bash
python build_tier1_unknown_shock.py \
  --returns   ../output/experiments/run_00_market_proxy/returns_run_00_market_proxy.csv \
  --shock     ../data/macro/shock_indices.csv \
  --fedsig    ../data/fomc_statements/fedsignal.csv \
  --train_end 2017-12-31 \
  --k_u 8 \
  --out ../output/experiments/run_00_market_proxy/tier1_unknown_shock_run_00_market_proxy.csv
```

### 2.4 Tier 2: OCC channels + ShockLedger export

```bash
python build_tier2_occ.py \
  --tier1     ../output/experiments/run_00_market_proxy/tier1_unknown_shock_run_00_market_proxy.csv \
  --returns   ../output/experiments/run_00_market_proxy/returns_run_00_market_proxy.csv \
  --shock     ../data/macro/shock_indices.csv \
  --train_end 2017-12-31 \
  --smooth_win 3 \
  --q_nn 0.90 \
  --out_channels  ../output/experiments/run_00_market_proxy/tier2_channels_run_00_market_proxy.csv \
  --out_ledger_y  ../output/experiments/run_00_market_proxy/tier2_shock_ledger_run_00_market_proxy.yaml \
  --out_ledger_p  ../output/experiments/run_00_market_proxy/tier2_shock_ledger_run_00_market_proxy.csv \
  --out_macro_sig ../output/experiments/run_00_market_proxy/macro_signature_run_00_market_proxy.csv
```

### 2.5 Build mapping from ShockLedger (macro → objectives)

```bash
python make_mapping_from_ledger.py \
  --ledger    ../output/experiments/run_00_market_proxy/tier2_shock_ledger_run_00_market_proxy.yaml \
  --panel     ../data/macro/shock_indices.csv \
  --fit_end   2017-12-31 \
  --train_end 2017-12-31 \
  --scaling maha --s0 1.0 \
  --regularize_weights 0.1 \
  --out ../output/experiments/run_00_market_proxy/mapping_from_ledger_run_00_market_proxy.yaml
```

### 2.6 Generate scenario paths (ABM bridge)

```bash
python abm_bridge.py \
  --panel   ../data/macro/shock_indices.csv \
  --mapping ../output/experiments/run_00_market_proxy/mapping_from_ledger_run_00_market_proxy.yaml \
  --out_dir ../output/experiments/run_00_market_proxy/scenarios_run_00_market_proxy_seed_0/ \
  --auto_select \
  --start_from 2009-05-12 --end_date 2023-12-31 --fit_end 2017-12-31 \
  --select_objectives risk_off,risk_on,hawkish,dovish \
  --select_top_k 3 --select_horizon_days 5 --select_var_alpha 10 \
  --scale_mode constant --scale_to_score 1.0 \
  --n_paths 1000 --ridge_alpha 10 \
  --seed 0
```

### 2.7 Rolling online mapper (rolling shock/context outputs)

```bash
python rolling_mapper_online.py \
  --channels    ../output/experiments/run_00_market_proxy/tier2_channels_run_00_market_proxy.csv \
  --base_shocks ../data/macro/shock_indices.csv \
  --returns     ../output/experiments/run_00_market_proxy/returns_run_00_market_proxy.csv \
  --window 126 --lambda_decay 0.995 --alpha 0.001 --l1_ratio 0.5 \
  --outdir ../output/experiments/run_00_market_proxy/shocks_run_00_market_proxy
```

SCR construction hyperparameters are fixed (no tuning within SCR build scripts).


### 2.8 Seeds / repetitions

* **Non market-proxy universes:** run `seed 0` and `seed 1`.
* **Market-proxy universes:** use the random seeds generated by:

  * `code/src/seed_generation.ipynb`

---

## 3) Counterfactual Continuation for Critic Target Augmentation

From `code/src/ccm/`, run:

```bash
python run_ccm.py --config ../../output/configs/config_seed_0.yaml
```

To run other universes, update the run/universe name inside the config accordingly.
Experimental artifacts are provided to support research transparency and reproducibility.

---

## 4) Experiment variants

The config provided in `code/output/configs/` corresponds to SCR-PPO-Full. To run the other variants, update `output:dir` with the corresponding suffix:

* **_no_cf (SCR-PPO-NoCF):** set `beta_cf = 0.0`
* **_cf_2_half :** set `beta_cf = 0.25`
* **_cf_7_half :** set `beta_cf = 0.75`
* **_cf_10 :** set `beta_cf = 1.0`
* **_vanilla (SCR-PPO-RewardOnly):** set
  `lambda_rho`, `beta_cf`, `graph:kappa`, `rl:caps:kappa_smooth`, `rl:laplacian_gamma` to `0`

---

## 5) Aggregate results

After all runs finish, execute:

* `code/src/output/experiments/res.ipynb`
* `code/src/output/experiments/run_00_market_proxy/diag_plot.ipynb`

to produce the final performance summaries and diagnostic plots.


## Third-party Resources

This repository may refer to third-party datasets, pretrained models, software libraries, and external repositories. These resources remain governed by their own licenses and terms.

Original third-party datasets are not redistributed in this repository. Users must obtain any required third-party datasets and external resources from the relevant original providers and comply with their applicable licenses and terms of use.

The repository license applies only to original code and materials authored for this repository.


## License, Copyright, and Citation

Copyright (c) 2026 Vanya Priscillia Bendatu and Yao Lu.

This repository is made publicly available to support transparency and reproducibility of the associated paper.
 
Original code and materials authored by the repository authors are licensed under the PolyForm Noncommercial License 1.0.0, to the extent those materials are owned or licensable by the authors. Commercial use is not permitted under this repository license. The full license text is available in the `LICENSE.txt` file.

Third-party datasets, pretrained models, libraries, and external resources are not redistributed or relicensed by this repository. Users are responsible for obtaining such materials from their original providers and complying with their respective licenses and terms of use.

If you use this repository or its materials in any publication, report, benchmark, comparison, reproduction study, or derivative research work, please cite the associated paper:

```bibtex
@inproceedings{bendatu2026scr,
  title     = {Reinforcement Learning with Scenario-Context Rollout in Portfolio Management},
  author    = {Bendatu, Vanya Priscillia and Lu, Yao},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  doi       = {10.1145/3770855.3817999}
}
```
