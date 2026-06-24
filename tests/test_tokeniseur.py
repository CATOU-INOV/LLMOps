import pytest
from unittest.mock import MagicMock, patch
from src.tokeniseur import tokeniser, compter_tokens, decoder


@pytest.fixture
def tokeniseur_mock():
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3, 4, 5]
    tok.decode.return_value = "bonjour le monde"
    return tok


def test_tokeniser_retourne_liste(tokeniseur_mock):
    ids = tokeniser("bonjour", tokeniseur_mock)
    assert isinstance(ids, list)
    assert len(ids) > 0
    tokeniseur_mock.encode.assert_called_once_with("bonjour", add_special_tokens=True)


def test_compter_tokens(tokeniseur_mock):
    nb = compter_tokens("bonjour", tokeniseur_mock)
    assert nb == 5


def test_compter_tokens_texte_vide(tokeniseur_mock):
    tokeniseur_mock.encode.return_value = []
    nb = compter_tokens("", tokeniseur_mock)
    assert nb == 0


def test_decoder(tokeniseur_mock):
    resultat = decoder([1, 2, 3], tokeniseur_mock)
    assert resultat == "bonjour le monde"
    tokeniseur_mock.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)
