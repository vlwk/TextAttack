from textattack.models.wrappers import ModelWrapper
from typing import List
import torch
import argparse

class FairseqEnFrWrapper(ModelWrapper):
    """
    A wrapper for the fairseq WMT'14 English-French model that handles
    multiprocessing by using a class-level model cache.
    """
    _model_cache = None  # Class-level model cache

    def __init__(self, model=None):
        """Initialize the wrapper. The model can be passed in or created on-demand."""
        if model is not None:
            FairseqEnFrWrapper._model_cache = model
        self.model = None

    def _get_model(self):
        """Get or create the model on demand in the worker process."""
        if self.model is None:
            if FairseqEnFrWrapper._model_cache is not None:
                self.model = FairseqEnFrWrapper._model_cache
            else:
                # Only add safe globals if we're creating a new model
                torch.serialization.add_safe_globals([argparse.Namespace])
                self.model = torch.hub.load(
                    'pytorch/fairseq',
                    'transformer.wmt14.en-fr',
                    tokenizer='moses',
                    bpe='subword_nmt',
                    weights_only=False,
                    verbose=False
                ).eval()
                FairseqEnFrWrapper._model_cache = self.model
        return self.model

    def __call__(self, text_input_list: List[str]) -> List[str]:
        """
        Args:
            input_texts: List[str]
        
        Return:
            ret: List[str]
                Result of translation. One per element in input_texts.
        """
        model = self._get_model()
        return [model.translate(text) for text in text_input_list]
