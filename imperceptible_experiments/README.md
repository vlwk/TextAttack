Folder structure:

imperceptible_experiments

    attack_scripts
        machine_translation

        named_entity_recognition
        sentiment_classification
            dair_ai_emotion
                bhadresh_distilbert_base_uncased_emotion.py
                gpt4.py
        textual_entailment
        toxic_content_detection

    model_wrappers
        machine_translation
            fairseq_en_fr.py
            gpt4.py
        named_entity_recognition
            dbmdz_bert_large_cased_finetuned_conll03_english.py
        sentiment_classification
            bhadresh_distilbert_nbase_uncased_emotion.py
            gpt4.py
        textual_entailment
            fairseq_roberta_mnli.py
        toxic_content_detection
            ibm_max_toxic.py

    results
        machine_translation
        named_entity_recognition
        sentiment_classification
            dair_ai_emotion
                num{args.num_examples}
                    bhadresh_distilbert_base_uncased_emotion
                    gpt4
        textual_entailment
        toxic_content_detection
