import numpy as np
from supersvd import supersvd
import matplotlib.pyplot as plt

ts = np.fromfile('ts.std', dtype=np.float32).reshape(1147, 28, 31)
ps = np.fromfile('ps.std', dtype=np.float32).reshape(1147, 28, 31)

svd = supersvd(ts, ps, 4)

first_mode_ts = svd.x_vect[0]

plt.matshow(first_mode_ts)
plt.savefig('first_mode_ts.png')

first_ps_timeseries = svd.x_coeff[0]

plt.figure()
plt.plot(first_ps_timeseries)
plt.savefig('first_ps_timeseries.png')

print('Correlation coefficient between first_ts_timeseries and first_ps_timeseries:', svd.corrcoeff[0])



