#!/bin/bash

for perturb in homoglyphs deletions reorderings invisible
do
  python attack_scripts/ner_t/conll2003/bertconll.py \
    --perturbs_start_incl 1 \
    --perturbs_end_excl 6 \
    --perturbation_type $perturb \
    --target_suffix PER \
    --popsize 5 \
    --maxiter 3 \
    --num_examples 50
done
