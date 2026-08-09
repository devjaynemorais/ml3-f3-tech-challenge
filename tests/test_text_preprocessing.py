from src.features.text_preprocessing import clean_text


def test_clean_text_normaliza_minusculas_acentos_e_espacos() -> None:
    resultado = clean_text("  Dor TORÁCICA   Intensa!")
    assert resultado == "dor toracica intensa!"


def test_clean_text_string_vazia() -> None:
    assert clean_text("") == ""
