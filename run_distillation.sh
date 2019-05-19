python2 distillation_main.py \
	--save-dir ../distill_results \
	--teacher-checkpoint models/mosaic.pth \
	--learning-rate 0.01 \
	--transfer-learning
	#--coco \
	#--coco-dataset data/COCO_2017/ \
	
	#--arch vgg11 \
	#--evaluate \
	#--slim-checkpoint /home/${user}/distill_results/checkpoint_86.tar

