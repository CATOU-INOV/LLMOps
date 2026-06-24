from transformers import AutoTokenizer, PreTrainedTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def charger_tokeniseur(nom_modele: str = DEFAULT_MODEL) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(nom_modele)


def tokeniser(texte: str, tokeniseur: PreTrainedTokenizer) -> list[int]:
    return tokeniseur.encode(texte, add_special_tokens=True)


def compter_tokens(texte: str, tokeniseur: PreTrainedTokenizer) -> int:
    return len(tokeniser(texte, tokeniseur))


def decoder(ids: list[int], tokeniseur: PreTrainedTokenizer) -> str:
    return tokeniseur.decode(ids, skip_special_tokens=True)
