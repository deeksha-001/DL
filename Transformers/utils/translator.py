from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.language_config import MODEL_CONFIG

loaded_models = {}

def load_model(model_name):

    if model_name not in loaded_models:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        loaded_models[model_name] = (
            tokenizer,
            model
        )

    return loaded_models[model_name]


def translate_text(text, language):

    config = MODEL_CONFIG[language]

    model_name = config["model"]

    tokenizer, model = load_model(model_name)

    if language == "French":

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        outputs = model.generate(**inputs)

        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    elif language == "Telugu":

        tokenizer.src_lang = "eng_Latn"

        inputs = tokenizer(
            text,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                "tel_Telu"
            )
        )

        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )