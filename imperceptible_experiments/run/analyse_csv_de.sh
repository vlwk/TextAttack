#!/bin/bash

# Expts 1a, 1d, 1e: MT-u

for model in fairseq_en_fr gemini2f gpt4; do
  for perturb in deletions homoglyphs invisible reorderings; do
    for budget in 1 5; do
      echo "Running with model=$model, perturb=$perturb, budget=$budget"
      python analysis/analyse_csv/mt_u/stats.py \
        --dataset_name wmt14enfr \
        --num_examples 50 \
        --model_name "$model" \
        --popsize 5 \
        --maxiter 3 \
        --perturbation_type "$perturb" \
        --budget "$budget"
    done
  done
done

# Expt 2a: TC-t

for perturb in deletions homoglyphs invisible reorderings; do
  for budget in 1 2 3 4 5; do
    echo "Running with model=$model, perturb=$perturb, budget=$budget"
    python analysis/analyse_csv/tc_t/stats.py \
      --dataset_name wikipedia_detox \
      --num_examples 50 \
      --model_name ibm_maxtoxic \
      --popsize 5 \
      --maxiter 3 \
      --perturbation_type "$perturb" \
      --budget "$budget"
  done
done

# Expt 3a: TE-t

for perturb in deletions homoglyphs invisible reorderings; do
  for budget in 1 2 3 4 5; do
    echo "Running with model=$model, perturb=$perturb, budget=$budget"
    python analysis/analyse_csv/te_t/stats.py \
      --dataset_name mnli \
      --num_examples 50 \
      --model_name fairseq_roberta_mnli \
      --popsize 5 \
      --maxiter 3 \
      --perturbation_type "$perturb" \
      --budget "$budget"
  done
done

# Expt 4a: NER-t

for PERTURB in homoglyphs deletions invisible reorderings; do
  for BUDGET in {1..5}; do
    echo "Running: $PERTURB with budget $BUDGET"
    python analysis/analyse_csv/ner_t/stats.py \
      --dataset_name "conll2003" \
      --num_examples 50 \
      --model_name "bertconll" \
      --popsize 5 \
      --maxiter 3 \
      --perturbation_type "$PERTURB" \
      --target_suffix "PER" \
      --budget "$BUDGET"
  done
done

# Expts 5a-5f: SC-t

for model in gpt4o gemini2f bhadresh_distilbert bhadresh_ft bhadresh_ft_enc t5_bhadresh; do
  for perturb in homoglyphs deletions invisible reorderings; do
    for budget in 1 5; do
      echo "Running: model=$model, perturb=$perturb, budget=$budget"
      python analysis/analyse_csv/sc_t/stats.py \
        --dataset_name dair_ai_emotion \
        --num_examples 50 \
        --model_name "$model" \
        --popsize 5 \
        --maxiter 3 \
        --perturbation_type "$perturb" \
        --target_class 1 \
        --budget "$budget"
    done
  done
done

# Expts 5g-5l: SC-o

for model in gpt4o gemini2f bhadresh_distilbert bhadresh_ft bhadresh_ft_enc t5_bhadresh; do
  for perturb in homoglyphs deletions invisible reorderings clean; do
    echo "Running: $model on $perturb"
    python analysis/analyse_csv/sc_o/stats.py \
      --dataset_name "emotion_perturbed_test" \
      --model_name "$model" \
      --perturbation_type "$perturb"
  done
done
