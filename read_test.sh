# run test
python read.py \
    --batch_size 8 \
    --load_dir read1 \
    --output_dir read2 \
    --dev_data_path './Data/HybridQA/test.row.json' \
    --predict_save_path './Data/HybridQA/test_answers_local.json'\
    --is_train 0 \
    --is_test 1 
    



