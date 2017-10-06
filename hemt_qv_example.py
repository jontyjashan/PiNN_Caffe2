from ac_qv_api import ACQVModel, plot_iv, predict_qs
import pinn.parser as parser
import pinn.preproc as preproc
import pinn.exporter as exporter
import pinn.deembed as deembed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

data_arrays = deembed.deembed(parser.read_s_par_mdm, './HEMT_bo/s_at_f_vs_VgVd.mdm', 1e-12, 1e-12, 1, 2)

#voltage is vg, vd
voltage = np.concatenate(
	(np.expand_dims(data_arrays[0], axis=1),
	 np.expand_dims(data_arrays[1], axis=1)), 
	axis=1
)

capas = np.array(data_arrays[6])

scale, vg_shift = preproc.compute_ac_meta(voltage, capas)

preproc_param = {
	'scale': scale,
	'vg_shift': vg_shift,
}


ac_model = ACQVModel('ac_model', input_dim=2, output_dim=1)
ac_model.add_data('train', [voltage, capas], preproc_param)
ac_model.build_nets([10, 10, 10], batch_size = 1275)
ac_model.train_with_eval(num_epoch = 10000, report_interval = 0)
ac_model.draw_nets() 
# ac_model.plot_loss_trend()

pred_qs, ori_qs, pred_grad, ori_grad = ac_model.predict_qs(
	voltage)

# Plot predicted q
plot_iv(
	voltage[:, 0], voltage[:, 1], ori_qs
)
# Plot dqdvg, compare original and predicted
plot_iv(
	voltage[:, 0], voltage[:, 1], ori_grad[:, 0],
	voltage[:, 0], voltage[:, 1], capas[:, 0]
)
# Plot dqdvd, compare original and predicted
plot_iv(
	voltage[:, 0], voltage[:, 1], ori_grad[:, 1],
	voltage[:, 0], voltage[:, 1], capas[:, 1]
)


