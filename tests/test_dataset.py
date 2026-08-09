from scripts.generate_synthetic_dataset import generate_dataset
from src.data.make_dataset import split_dataset


def test_generate_dataset_tem_colunas_e_classes_esperadas() -> None:
    df = generate_dataset(n_samples=30, seed=1)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"].unique()) == {"normal", "atencao", "urgente"}
    assert len(df) == 30


def test_generate_dataset_e_deterministico_por_seed() -> None:
    df1 = generate_dataset(n_samples=20, seed=7)
    df2 = generate_dataset(n_samples=20, seed=7)
    assert df1["text"].tolist() == df2["text"].tolist()


def test_split_dataset_preserva_o_total_de_amostras() -> None:
    df = generate_dataset(n_samples=90, seed=1)
    train, val, test = split_dataset(
        df, label_col="label", test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(train) + len(val) + len(test) == len(df)
    # test_size=0.2 e val_size=0.1 -> proporções aproximadas (arredondamento)
    assert abs(len(test) - 18) <= 2
    assert abs(len(val) - 9) <= 2


def test_split_dataset_estratifica_por_label() -> None:
    df = generate_dataset(n_samples=90, seed=1)
    train, _val, _test = split_dataset(
        df, label_col="label", test_size=0.2, val_size=0.1, random_state=42
    )
    assert set(train["label"].unique()) == {"normal", "atencao", "urgente"}
