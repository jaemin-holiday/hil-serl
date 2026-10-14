# REQUIRED: pass the demo pkl recorded by record_demos.py, e.g.
#   bash run_learner.sh --demo_path=demo_data/rby1_reach_20_demos_<ts>.pkl
# (the learner hard-asserts --demo_path is set)
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=rby1_reach \
    --checkpoint_path=../../experiments/rby1_reach/debug \
    --learner \
