# Controlled follow-up measurements

Median [minimum, maximum] from three runs in one batch. Failed-run time is consumed work, not time to solution. Peak memory is whole-process working set. Logs are preserved.

|Case|Method|Success|Seconds [range]|MiB|SSN|Loss calls|Fallbacks|
|---|---|---:|---|---:|---:|---:|---:|
|s4000|FBE-LBFGS|3/3|1.806 [1.384, 2.426]|134.0|0|281|0|
|s4000|P-CONT-D|3/3|3.459 [2.200, 4.178]|134.0|700|152|0|
|s4000|R-FISTA|3/3|0.977 [0.872, 1.286]|133.9|0|188|0|
|s16000|FBE-LBFGS|3/3|8.540 [7.879, 9.566]|243.8|0|341|0|
|s16000|P-CONT-D|3/3|22.152 [19.059, 23.393]|243.9|814|290|2|
|s16000|R-FISTA|3/3|4.621 [4.322, 6.193]|243.9|0|162|0|
|weak4000|FBE-LBFGS|3/3|1.466 [1.266, 1.763]|134.0|0|253|0|
|weak4000|P-CONT-D|3/3|4.420 [4.181, 7.747]|133.8|1388|110|0|
|weak4000|R-FISTA|3/3|0.873 [0.835, 1.016]|133.8|0|130|0|
|group8000|FBE-LBFGS|3/3|2.355 [1.783, 2.418]|170.5|0|134|0|
|group8000|P-CONT-D|3/3|3.700 [2.480, 3.855]|170.4|252|128|0|
|group8000|R-FISTA|3/3|1.486 [1.455, 1.887]|170.6|0|91|0|

All original gaps were recalculated with a separate explicit dual formula. Logged threshold margins were checked arithmetically; intermediate vectors were not retained, so this is not an independent trajectory or interval-arithmetic proof. Complete operation counts are in aggregate.json and the terminal JSON records.
