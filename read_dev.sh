# run dev
python read.py \
    --batch_size 8 \
    --load_dir read1 \
    --output_dir read2 \
    --dev_data_path './Data/HybridQA/dev.row.json' \
    --predict_save_path './Data/HybridQA/dev_answers.json'\
    --is_train 0 \
    --is_test 0