# Data Files

`offline_trials.csv` contains 1,512 target-resolution trials. Rows include the detector candidate count used by the target-resolution model.

`online_trials.csv` contains 228 robot trials. Online rows include the detector candidate count, whether robot execution was attempted, physical grasp success, full task success, and whether a wrong-target execution was prevented.

`instruction_pools/` stores per-scene instruction pools grouped by instruction type. Each entry contains only an index and instruction text.

`scenes/` stores observation images available for the released scenes. The online observation images for `scene_05` and `scene_20` are unavailable; their trial-level records and instruction pools are retained in `online_trials.csv` and `instruction_pools/online_instruction_pool.json`.

`calibration/threshold_sweep.csv` stores detector threshold sweep points used to recreate the calibration figure.
