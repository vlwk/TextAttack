#!/bin/bash

PERTURB_TYPES=("invisible" "reorderings" "deletions" "homoglyphs")
START_INDICES=(1 5)

for START in "${START_INDICES[@]}"; do
  END=$((START + 1))
  for PERTURB in "${PERTURB_TYPES[@]}"; do
    python attack_scripts/sc_t/dair_ai_emotion/gpt4o.py \
      --perturbs_start_incl "$START" \
      --perturbs_end_excl "$END" \
      --perturbation_type "$PERTURB" \
      --target_class 1 \
      --popsize 5 \
      --maxiter 3 \
      --num_examples 50
  done
done
