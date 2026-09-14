# Controlled follow-up measurements

Median [minimum, maximum] from three runs in one batch. Failed-run time is consumed work, not time to solution. Peak memory is whole-process working set. Logs are preserved.

|Case|Method|Success|Seconds [range]|MiB|SSN|Loss calls|Fallbacks|
|---|---|---:|---|---:|---:|---:|---:|
|s4000|P-CONT|3/3|2.507 [2.059, 4.130]|134.1|700|248|0|
|s4000|P-CONT-C|3/3|3.761 [3.146, 4.163]|134.0|700|200|0|
|s4000|P-CONT-D|3/3|2.567 [1.874, 4.199]|134.0|700|152|0|
|s4000|PQN|3/3|4.221 [4.100, 4.994]|134.2|1547|142|0|
|s4000|R-FISTA|3/3|0.951 [0.770, 0.996]|134.2|0|188|0|
|s4000|PN-MF|3/3|1.695 [1.643, 2.542]|133.9|0|13|0|
|s16000|P-CONT|3/3|18.562 [16.613, 21.568]|244.0|813|469|1|
|s16000|P-CONT-C|3/3|18.265 [14.939, 18.793]|243.9|805|378|1|
|s16000|P-CONT-D|3/3|16.074 [15.624, 20.306]|243.9|814|287|1|
|s16000|PQN|0/3|45.010 [45.001, 45.010]|244.1|3651|100|0|
|s16000|R-FISTA|3/3|3.720 [3.413, 4.080]|244.1|0|162|0|
|s16000|PN-MF|3/3|10.760 [9.438, 11.984]|244.1|0|13|0|
|weak4000|P-CONT|3/3|4.227 [3.619, 5.134]|134.2|1388|178|0|
|weak4000|P-CONT-C|3/3|4.857 [3.456, 4.996]|134.2|1388|144|0|
|weak4000|P-CONT-D|3/3|4.597 [4.219, 4.830]|134.1|1388|110|0|
|weak4000|PQN|3/3|0.833 [0.752, 1.024]|134.1|132|103|0|
|weak4000|R-FISTA|3/3|0.696 [0.640, 0.728]|134.3|0|130|0|
|weak4000|PN-MF|3/3|2.597 [2.571, 2.982]|134.1|0|16|0|
|group8000|P-CONT|3/3|3.105 [2.952, 3.250]|170.6|252|208|0|
|group8000|P-CONT-C|3/3|3.115 [2.956, 3.551]|170.7|252|168|0|
|group8000|P-CONT-D|3/3|2.881 [2.796, 2.999]|170.5|252|128|0|
|group8000|PQN|3/3|2.319 [2.317, 2.536]|170.7|220|100|0|
|group8000|R-FISTA|3/3|1.240 [1.144, 1.350]|170.9|0|91|0|
|group8000|PN-MF|3/3|3.922 [3.702, 3.992]|170.7|0|13|0|

All original gaps were recalculated with a separate explicit dual formula. Logged threshold margins were checked arithmetically; intermediate vectors were not retained, so this is not an independent trajectory or interval-arithmetic proof. Complete operation counts are in aggregate.json and the terminal JSON records.
