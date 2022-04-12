python exp_runner.py --mode train --conf ./confs/womask_satellite.conf  --gpu 0 --case jax \
                     --data_dir /share/phoenix/nfs04/S7/IARPA-SMART/delivery/SatelliteSfM/examples/outputs

python exp_runner.py --mode validate_mesh --conf ./confs/womask_satellite.conf --gpu 0 --case jax --is_continue \
                     --render_split_tag path_zoom \
                     --data_dir /share/phoenix/nfs04/S7/IARPA-SMART/delivery/SatelliteSfM/examples/outputs_path_zoom

python exp_runner.py --mode validate_mesh --conf ./confs/womask_satellite.conf --gpu 0 --case jax \
                     --render_split_tag path_zoom \
                     --data_dir /share/phoenix/nfs04/S7/IARPA-SMART/delivery/SatelliteSfM/examples/outputs_path_zoom