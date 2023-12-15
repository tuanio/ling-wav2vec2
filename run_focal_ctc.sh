for alpha in 0.99 0.75 0.5 0.25; do
    for gamma in 0.5 1 2; do
        accelerate launch --main_process_port 12345 finetune_w2v2_focal_ctc_linguistic.py --focal-alpha $alpha --focal-gamma $gamma
    done
done 