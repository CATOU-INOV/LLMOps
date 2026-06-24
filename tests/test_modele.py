import pytest
from unittest.mock import MagicMock, patch
import torch
from src.modele import generer


@pytest.fixture
def modele_et_tokeniseur():
    tokeniseur = MagicMock()
    tokeniseur.eos_token_id = 2
    # Simule inputs : input_ids de longueur 5
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokeniseur.return_value = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    tokeniseur.decode.return_value = "Nous livrons sous 48h."

    modele = MagicMock()
    # Simule outputs : 5 tokens prompt + 3 tokens générés
    modele.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])

    return modele, tokeniseur


def test_generer_retourne_chaine(modele_et_tokeniseur):
    modele, tokeniseur = modele_et_tokeniseur
    reponse = generer("Question test ?", modele, tokeniseur)
    assert isinstance(reponse, str)


def test_generer_appelle_generate(modele_et_tokeniseur):
    modele, tokeniseur = modele_et_tokeniseur
    generer("Question test ?", modele, tokeniseur, nb_tokens_max=50)
    modele.generate.assert_called_once()
    call_kwargs = modele.generate.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 50


def test_generer_decode_seulement_nouveaux_tokens(modele_et_tokeniseur):
    modele, tokeniseur = modele_et_tokeniseur
    generer("Question test ?", modele, tokeniseur)
    ids_decodes = tokeniseur.decode.call_args[0][0]
    assert ids_decodes.tolist() == [10, 11, 12]
