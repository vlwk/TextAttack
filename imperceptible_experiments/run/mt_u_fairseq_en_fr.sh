#!/bin/bash

PERTURB_TYPES=("homoglyphs" "deletions" "reorderings" "invisible")

for PERTURB in "${PERTURB_TYPES[@]}"; do
  python attack_scripts/mt_u/wmt14enfr/fairseq_en_fr.py \
    --perturbs_start_incl 1 \
    --perturbs_end_excl 6 \
    --perturbation_type "$PERTURB" \
    --popsize 5 \
    --maxiter 3 \
    --num_examples 50
done
