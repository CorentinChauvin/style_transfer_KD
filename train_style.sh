python2 neural_style/neural_style.py train \
  --style-image images/style-images/mosaic.jpg \
  --dataset data/COCO_2017/ \
  --save-model-dir ../training_results/ \
  --batch-size 4 \
  --checkpoint-interval 100 \
  --log-interval 10 \
  --cuda 1 \
  --epochs 3 \
  
#--model saved_models/mosaic.pth \
