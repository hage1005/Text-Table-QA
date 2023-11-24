# run dev
torchrun --standalone --nproc-per-node=gpu read.py \
    --batch_size 8 \
    --load_dir read1 \
    --output_dir read2 \
    --is_train 1 \
    --is_test 0 \
    --learning_rate 1e-5