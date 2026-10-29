# Source data contract

The loader reads a ZIP member as an Excel workbook, converting cells to strings. Required columns include 受理号, 药品名称, 承办日期, 审结日期, 审评结论, 企业名称, 申请事项, 剂型, 药品类型（CDE）, 注册分类（CDE）, 申请类型（CDE） and 创新类型 (the latter is audited, not used as a predictor). The actual feature whitelist and outcome strings are in config.example.json. The original final source audit accepts ISO YYYY-MM-DD date strings and reconstructs elapsed calendar days independently.

The study restricts eligibility to production/import applications. Pending outcomes are censored only when completion is unrecorded; ambiguous labels or inconsistent dates are excluded. See prepare.py for exact precedence. These exclusions depend on post-acceptance outcomes and do not define an unselected prospective cohort.

Training ends 2020-12-31, validation 2022-12-31, test acceptance 2024-12-31. With observation_end=null, preparation infers the observation cutoff from the latest eligible date; for the source snapshot this was 2026-05-19. This is not a verified extraction/last-contact date. The final approval audit intentionally fixes that paper-specific date.

Approval counts use t < horizon, not t <= horizon. Exact source-date reconciliation verifies processing consistency but cannot establish true follow-up, publication times or whether source dates were backfilled.

Only load trusted pickle/model artifacts. Original cached objects were analyzed with pandas 3.0.5 locally; use the generating environment when loading historical serialized objects, or regenerate the cohort from an authorized source. A fresh cohort does not guarantee numerical equivalence to archived models.

Author-provided source portals are listed in [DATA_SOURCES.md](../DATA_SOURCES.md). No records or aggregate outputs from these sources are distributed.
