python2 neural_style/neural_style.py eval \
  --content-image images/content-images/amber.jpg \
  --model ../distill_results/checkpoint_5.tar \
  --output-image images/output-images/test.jpg \
  --cuda 0 \
  \
  --print-loss \
  --style-image images/style-images/mosaic.jpg \
  --distilled \
  
#--model models/mosaic.pth \
#--model ../distill_results/checkpoint_0.tar \
