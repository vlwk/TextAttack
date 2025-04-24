from textattack.trainer import Trainer
import torch

class Seq2SeqTrainer(Trainer):
    def get_train_dataloader(self, dataset, adv_dataset, batch_size):
        def collate_fn(data):
            input_texts = []
            target_texts = []
            is_adv_sample = []
            for item in data:
                if "_example_type" in item[0].keys():
                    item[0].pop("_example_type")
                    _input, label = item
                    is_adv_sample.append(True)
                else:
                    _input, label = item
                    is_adv_sample.append(False)

                if isinstance(_input, dict):
                    _input = list(_input.values())[0]
                input_texts.append(_input)
                target_texts.append(label)

            return input_texts, target_texts, torch.tensor(is_adv_sample)

        if adv_dataset:
            dataset = torch.utils.data.ConcatDataset([dataset, adv_dataset])

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def training_step(self, model, tokenizer, batch):
        input_texts, target_texts, is_adv_sample = batch
        device = textattack.shared.utils.device

        # Tokenize inputs and targets
        model_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).input_ids
        labels = labels.to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass
        outputs = model(**model_inputs, labels=labels)
        loss = outputs.loss

        # No preds/targets returned in this setting
        preds = torch.zeros(len(target_texts))  # dummy
        _targets = torch.zeros(len(target_texts))  # dummy

        return loss, preds, _targets
