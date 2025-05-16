import pandas as pd
from transformers import PreTrainedTokenizerFast
from typing import List


class WordIndexAnnotator:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        if not tokenizer.is_fast:
            raise ValueError("Tokenizer must be a fast tokenizer to use `.word_ids()`")
        self.tokenizer = tokenizer

    def _get_token_indices_per_word(self, text: str, verbose=False) -> List[List[int]]:
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            truncation=True,
            padding=False
        )

        input_ids = encoding["input_ids"][0].tolist()
        word_ids = encoding.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        token_groups = []
        max_word_id = max([wid for wid in word_ids if wid is not None], default=-1)
        for word_idx in range(max_word_id + 1):
            indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            token_groups.append(indices)

        if verbose:
            self._print_token_debug_info(text, tokens, input_ids, word_ids)

        return token_groups

    def annotate(self, df: pd.DataFrame, verbose=False) -> pd.DataFrame:
        df = df.copy()
        df["word_indices_perturbed"] = df["input"].apply(
            lambda x: self._get_token_indices_per_word(x, verbose=verbose)
        )
        df["word_indices_clean"] = df["original_text"].apply(
            lambda x: self._get_token_indices_per_word(x, verbose=verbose)
        )
        return df

    def print_token_info(self, text: str):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            truncation=True,
            padding=False
        )
        input_ids = encoding["input_ids"][0].tolist()
        word_ids = encoding.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        self._print_token_debug_info(text, tokens, input_ids, word_ids)

    def _print_token_debug_info(self, text: str, tokens: List[str], input_ids: List[int], word_ids: List[int]):
        print(f"\nFull token sequence for: \"{text}\"\n")
        for i, (token, tid, wid) in enumerate(zip(tokens, input_ids, word_ids)):
            label = f"word {wid}" if wid is not None else "SPECIAL"
            print(f"[{i:2}] {token:<12} (id: {tid}) ← {label}")
