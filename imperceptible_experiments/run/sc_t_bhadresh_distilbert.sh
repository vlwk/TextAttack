#!/bin/bash

PERTURB_TYPES=("invisible" "reorderings" "deletions" "homoglyphs")

for PERTURB in "${PERTURB_TYPES[@]}"; do
  python attack_scripts/sc_t/dair_ai_emotion/bhadresh_distilbert.py \
    --perturbs_start_incl 1 \
    --perturbs_end_excl 6 \
    --perturbation_type "$PERTURB" \
    --target_class 1 \
    --popsize 5 \
    --maxiter 3 \
    --num_examples 50
done
