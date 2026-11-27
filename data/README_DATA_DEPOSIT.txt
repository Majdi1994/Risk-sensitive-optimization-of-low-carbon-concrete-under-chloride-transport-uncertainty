ASTM C1556 modelling corpus deposit
=================================

Primary file: ASTM_C1556_n540.csv (n = 540 records from 25 sources).

Compressive strength: every modelling row has a finite 28-day strength value. Where a source record lacked strength and the binder system was ordinary Portland cement with fly ash and/or slag, the value was filled using an auxiliary model trained on the UCI concrete compressive-strength dataset. Silica-fume and metakaolin mixtures with missing strength were not filled by that model.

The column cs_provenance records whether compressive strength was taken from the source materials (`source_reported`) or filled with the UCI auxiliary model (`uci_filled`). Headline analyses in the paper use the full 540-record archive.
