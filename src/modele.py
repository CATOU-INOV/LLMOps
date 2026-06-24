from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def charger_modele(nom_modele: str = DEFAULT_MODEL) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(nom_modele)


def generer(
    prompt: str,
    modele: PreTrainedModel,
    tokeniseur: PreTrainedTokenizer,
    nb_tokens_max: int = 200,
) -> str:
    inputs = tokeniseur(prompt, return_tensors="pt")
    nb_tokens_prompt = inputs["input_ids"].shape[1]
    outputs = modele.generate(
        **inputs,
        max_new_tokens=nb_tokens_max,
        do_sample=False,
        pad_token_id=tokeniseur.eos_token_id,
    )
    tokens_generes = outputs[0][nb_tokens_prompt:]
    return tokeniseur.decode(tokens_generes, skip_special_tokens=True).strip()
