# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 5 --mask_feature_prob 0.001 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 5 --mask_feature_prob 0.001 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 5 --mask_feature_prob 0.008 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 5 --mask_feature_prob 0.008 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 10 --mask_feature_prob 0.001 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 10 --mask_feature_prob 0.001 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 10 --mask_feature_prob 0.008 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.025 --mask_time_length 10 --mask_feature_prob 0.008 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 5 --mask_feature_prob 0.001 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 5 --mask_feature_prob 0.001 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 5 --mask_feature_prob 0.008 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 5 --mask_feature_prob 0.008 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 10 --mask_feature_prob 0.001 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 10 --mask_feature_prob 0.001 --mask_feature_length 64
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 10 --mask_feature_prob 0.008 --mask_feature_length 16
# accelerate launch finetune_w2v2_only.py --mask_time_prob 0.05 --mask_time_length 10 --mask_feature_prob 0.008 --mask_feature_length 64

for i in {0..15}; do
    python finetune_w2v2_only.py --index $i
    bash run_md_d.sh "ablation_study/spec_augment"
done

# python finetune_w2v2_only.py --index 15
# bash run_md_d.sh "ablation_study/spec_augment"