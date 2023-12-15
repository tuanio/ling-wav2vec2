# echo "Public test"
python md_d_metric/align_data.py --subset public_test --out-file $1
python md_d_metric/ins_del_cor_sub_analysis.py --out-file $1

# echo "Private test"
python md_d_metric/align_data.py --subset private_test --out-file $1
python md_d_metric/ins_del_cor_sub_analysis.py --out-file $1

echo ================ >> $1