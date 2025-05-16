#!/bin/bash

PERTURB_TYPES=("homoglyphs" "deletions" "reorderings" "invisible")
START_INDICES=(1 5)

for START in "${START_INDICES[@]}"; do
  END=$((START + 1))
  for PERTURB in "${PERTURB_TYPES[@]}"; do
    python attack_scripts/mt_u/wmt14enfr/gemini2f.py \
      --perturbs_start_incl "$START" \
      --perturbs_end_excl "$END" \
      --perturbation_type "$PERTURB" \
      --popsize 5 \
      --maxiter 3 \
      --num_examples 50
  done
done
