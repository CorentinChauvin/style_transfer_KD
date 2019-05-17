python neural_style/neural_style.py eval \
  --content-image images/content-images/amber.jpg \
  --model ../distill_results/checkpoint_45.tar \
  --output-image images/output-images/test.jpg \
  --cuda 0 \
  --distilled
  
#--model saved_models/mosaic.pth \
