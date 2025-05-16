#!/bin/bash

for perturb in homoglyphs deletions reorderings invisible
do
  python attack_scripts/te_t/mnli/fairseq_roberta_mnli.py \
    --perturbs_start_incl 1 \
    --perturbs_end_excl 6 \
    --perturbation_type $perturb \
    --popsize 5 \
    --maxiter 3 \
    --num_examples 10
done
