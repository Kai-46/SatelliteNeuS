python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case kai/horse && python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case kai/horse --is_continue


python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case kai/bagel
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/kitty
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/duck
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/buddha
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/camera
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/pig
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/kettle
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case kai/dragon
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/sneaker
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/monk
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case fujun/head


python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case sai/dragon
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case sai/tree
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case sai/triton
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case sai/girl
python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case sai/pony

python exp_runner.py --mode train --conf ./confs/kai_womask_train.conf  --gpu 0 --case sai/triton_scaled

#### render test
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case kai/bagel --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/kitty --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/duck --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/buddha --is_continue &
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/camera --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/pig --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/kettle --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case kai/dragon --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/sneaker --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case fujun/monk --is_continue

python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case sai/triton_scaled --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case sai/girl --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case sai/tree --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case sai/pony --is_continue
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case sai/dragon --is_continue


#### render test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case kai/bagel --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/duck --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/camera --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/pig --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/kettle --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/sneaker --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/kitty --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case kai/dragon --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/buddha --is_continue --render_split_tag test_hard
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test_hard.conf --gpu 0 --case fujun/monk --is_continue --render_split_tag test_hard

#### render train
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_train.conf --gpu 0 --case fujun/kitty --is_continue



python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case kai/bagel --is_continue --render_split_tag test
python exp_runner.py --mode validate_mesh --conf ./confs/kai_womask_test.conf --gpu 0 --case kai/dragon --is_continue --render_split_tag test

