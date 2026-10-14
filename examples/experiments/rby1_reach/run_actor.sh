export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=rby1_reach \
    --checkpoint_path=../../experiments/rby1_reach/debug \
    --actor \
