# Imperceptible Experiments

This repository contains the implementation of imperceptible adversarial attacks on various NLP tasks using character-level perturbations. Based on the "Bad Characters" paper (Boucher et al., 2021), we implement and extend their approach to attack various NLP models including large language models and two fine-tuned Transformers based on the [Bhadresh](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) model.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up API keys (if using LLM-based models):

```bash
export OPENAI_API_KEY="your-key-here"  # For GPT-4
export GEMINI_API_KEY="your-key-here"  # For Gemini
```

## Project Structure

```
imperceptible_experiments/
├── run/                    # Entry point scripts for all experiments
│   ├── mt_u_*             # Machine translation (untargeted) experiments
│   ├── sc_t_*             # Sentiment classification (targeted) experiments
│   ├── te_t_*             # Textual entailment (targeted) experiments
│   ├── tc_t_*             # Toxic content detection (targeted) experiments
│   └── ner_t_*            # Named entity recognition (targeted) experiments
├── attack_scripts/         # Implementation of attack algorithms
├── model_wrappers/        # Model-specific wrapper implementations
├── datasets/              # Dataset loading and preprocessing
├── analysis/             # Analysis scripts and utilities
├── results/              # Experiment results and outputs
└── finetune_scripts/     # Model fine-tuning scripts
```

## Perturbation Types

We implement four types of imperceptible perturbations:

1. **Homoglyphs** (`homoglyphs`):

   - Replaces characters with visually similar Unicode characters
   - Example: 'a' → 'а' (Cyrillic a)

2. **Invisible Characters** (`invisible`):

   - Inserts zero-width and invisible Unicode characters
   - Example: "hello" → "he\u200Bllo" (zero-width space)

3. **Deletions** (`deletions`):

   - Uses delete control characters to visually remove characters
   - Example: "hello" → "he\u0008\u0008llo" (backspace)

4. **Reorderings** (`reorderings`):
   - Uses bidirectional text control characters to reorder text
   - Example: "hello" → "he\u202Ell\u202Co" (bidirectional override)

## Running Experiments

All experiments are launched through scripts in the `run/` directory. Each script follows the naming convention:

- `{task}_{mode}_{model}.sh`
  - task: mt (machine translation), sc (sentiment classification), etc.
  - mode: t (targeted) or u (untargeted)
  - model: specific model identifier

### Available Run Scripts:

Machine Translation:

- `mt_u_gpt4o.sh`: MT-u experiment on the GPT-4o model.
- `mt_u_gemini2f.sh`: MT-u experiment on the Gemini 2.0 Flash model.

Sentiment Classification:

- `sc_t_bhadresh_distilbert.sh`: SC-t experiment on the [Bhadresh](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) model.
- `sc_t_bhadresh_ft.sh`: SC-t experiment on the [Bhadresh-ft]() model.
- `sc_t_bhadresh_ft_enc.sh`: SC-t experiment on the [Bhadresh-ft-enc]() model.
- `sc_t_gpt4o.sh`: SC-t experiment on GPT-4o.
- `sc_t_gemini2f.sh`: SC-t experiment on Gemini 2.0 Flash.
- `sc_t_t5_bhadresh.sh`: SC-t on the [Bhadresh](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) model with [T5-ft] placed upstream of it.

Toxic Content Classification:

- `tc_t_ibm_maxtoxic.sh`: TC-t experiment on the IBM Max Toxic model.

Textual Entailment:

- `te_t_fairseq_roberta_mnli.sh`: TE-t experiment on the Fairseq RoBERTa MNLI model.

Named Entity Recognition:

- `ner_t_bertconll.sh`: NER-t experiment on the BERT-CoNLL model.

The scripts in `/run` call attack scripts in `/attack_scripts`.

Example: Machine Translation end-to-end

To initiate an attack, create a file `/attack_scripts/(task identifier)/(dataset_name)/(model_name).py`. You will need to create a model wrapper in `/model_wrappers` as well.

Here, we have created `attack_scripts/mt_u/wmt14enfr/gpt4o.py`.

Next, we create a model wrapper `model_wrappers/mt_u/fairseq_en_fr.py`.

We have defined an attack recipe in `TextAttack/textattack/attack_recipes/bad_characters_2021.py`. The model wrapper is passed into the attack recipe to derive an `Attack` object. Other parameters to the attack recipe include the `goal_function_type` and `perturbation_type`.

Next, you need to find a dataset and convert it to a `TextAttack.Dataset` object. We choose the WMT14 (en-fr) dataset.

Finally, we can pass both `Attack` and `Dataset` into `Attacker` to start the attack. You should save the results in `/results/(task identifier)/(dataset_name)/num(rows)/(model_name)/pop(popsize)_iter(maxiter)/(perturbation_type)/pert(num_perturbs)/log.csv`.

You may trigger the above using:

```bash
# Untargeted attack maximizing Levenshtein distance
./run/mt_u_gpt4o.sh --perturbs_start_incl 1 --perturbs_end_excl 6 \
    --perturbation_type "invisible" --popsize 5 --maxiter 3 --num_examples 50
```

### Parameters

Common parameters across scripts:

- `--perturbs_start_incl`: Starting number of perturbations (inclusive)
- `--perturbs_end_excl`: Ending number of perturbations (exclusive)
- `--perturbation_type`: Type of perturbation to use
- `--popsize`: Population size for differential evolution search
- `--maxiter`: Maximum iterations for differential evolution
- `--num_examples`: Number of examples to attack

Some tasks have task-specific parameters: see the attack recipe for details.

## Output Format

Results are saved in the following structure:

The CSV files contain:

- Original text
- Perturbed text
- Original model output
- Perturbed model output
- Number of queries used
- Success/failure status
- Attack parameters used

## Requirements

Core dependencies (see `requirements.txt` for full list):

- textattack>=0.3.7
- torch>=1.7.0
- transformers>=4.30.0
- datasets>=2.4.0
- pandas>=1.0.1
- numpy>=1.21.0
