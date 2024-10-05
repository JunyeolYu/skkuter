cd skkuter_op
TORCH_CUDA_ARCH_LIST="8.9" python3 setup.py install
cd ..
CUDA_VISIBLE_DEVICES=0 python3 test_script.py --model /home/model/ -b 2
