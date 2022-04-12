import os
import json

train_data_dir = '/share/phoenix/nfs04/S7/kz298/satellite_stuff/dfc2019_data_clean'

cmd_template = 'python exp_runner.py --mode validate_mesh --conf ./confs/womask_satellite.conf --gpu 0 --is_continue --render_split_tag path_superclose --case {case} --data_dir {data_dir} --n_latent_codes {n_latent_codes}'
base_dir = '/share/phoenix/nfs04/S7/kz298/satellite_stuff/dfc2019_data_clean_path_superclose'

fp = open('./cmds_path.sh', 'w')

for scene in os.listdir(base_dir):
    with open(os.path.join(train_data_dir, scene, 'cam_dict_norm.json')) as tmp_fp:
        cam_dict_norm = json.load(tmp_fp)

    cmd = cmd_template.format(case=scene, data_dir=os.path.join(base_dir, scene), n_latent_codes=str(len(cam_dict_norm)))

    fp.write(cmd + '\n\n')

fp.close()

