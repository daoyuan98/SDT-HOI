CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --world-size 4 \
    --print-interval 50 \
    --epochs 20 \
    --weight-decay 1e-4 \
    --lr-drop 10 \
    --batch-size 4 \
    --k 10 \
    --score_thres 0.2 \
    --encoder_layer 3 \
    --comp_layer    3 \
    --lr-head 2e-4 \
    --lr-decay 1e-4 \
    --dataset vcoco \
    --backbone resnet50 \
    --data-root vcoco/ \
    --partitions trainval test \
    --pretrained pretrained_models/detr/detr-r50-vcoco.pth \
    --output-dir checkpoints/SDT_vcoco 

