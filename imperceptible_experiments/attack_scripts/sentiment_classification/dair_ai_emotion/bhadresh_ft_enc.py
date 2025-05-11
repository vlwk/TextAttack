import textattack
from imperceptible_experiments.model_wrappers.sentiment_classification.bhadresh_ft_encoder_wrapper import BhadreshEncoderWrapper
import argparse
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from imperceptible_experiments.model_architectures.encoder_model import WordEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--perturbs_start_incl", type=int, required=True)
parser.add_argument("--perturbs_end_excl", type=int, required=True)
parser.add_argument("--perturbation_type", type=str, required=True, choices=["homoglyphs", "invisible", "deletions", "reorderings"])
parser.add_argument("--target_class", type=int, required=True, choices=range(6))
parser.add_argument("--popsize", type=int, default=5)
parser.add_argument("--maxiter", type=int, default=3)
parser.add_argument("--num_examples", type=int, required=True)

args = parser.parse_args()

assert args.perturbs_start_incl < args.perturbs_end_excl, "perturbs_start_incl must be less than perturbs_end_excl"
assert args.popsize > 0, "popsize must be positive"
assert args.maxiter > 0, "maxiter must be positive"
assert args.num_examples > 0, "num_examples must be positive"

# === Load fine-tuned encoder model ===
MODEL_DIR = "models/bhadresh_ft_enc/checkpoints"
MODEL_WEIGHTS = f"{MODEL_DIR}/model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
base_model = DistilBertModel.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = WordEncoder(base_model).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()

model_wrapper = BhadreshEncoderWrapper(model=model, tokenizer=tokenizer, device=device)

for pert in range(args.perturbs_start_incl, args.perturbs_end_excl):
    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper=model_wrapper,
        goal_function_type="targeted_bonus",
        perturbation_type=args.perturbation_type,
        target_class=args.target_class,
        perturbs=pert,
        popsize=args.popsize,
        maxiter=args.maxiter
    )

    dataset = textattack.datasets.HuggingFaceDataset("dair-ai/emotion", split="test")
    assert args.num_examples <= len(dataset), f"num_examples must be ≤ {len(dataset)}"

    checkpoint_dir = (
        f"results/sentiment_classification/dair_ai_emotion/"
        f"num{args.num_examples}/bhadresh_ft_enc/"
        f"pop{args.popsize}_iter{args.maxiter}/"
        f"{args.perturbation_type}/target{args.target_class}/pert{pert}"
    )

    log_to_csv = f"{checkpoint_dir}/log.csv"

    attack_args = textattack.AttackArgs(
        num_examples=args.num_examples,
        checkpoint_interval=10,
        checkpoint_dir=checkpoint_dir,
        log_to_csv=log_to_csv
    )

    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
