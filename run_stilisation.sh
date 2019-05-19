python2 neural_style/neural_style.py eval \
  --content-image images/content-images/amber.jpg \
  --model ../distill_results/checkpoint_0.tar \
  --output-image images/output-images/transfer_0.jpg \
  --cuda 0 \
  --distilled
  
#--model saved_models/mosaic.pth \
