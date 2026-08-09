"""Gera um dataset sintético de laudos médicos rotulados por urgência.

Não há dataset real versionado no repositório (ver README — seção Dataset
para instruções de como plugar um dataset real, ex.: Medical Abstracts TC
Corpus ou recortes do MIMIC-III). Este script produz um CSV plausível em
português (coluna `text` + coluna `label` em {normal, atencao, urgente}) só
para que o pipeline de treino/serving seja executável de ponta a ponta desde
o primeiro clone do repositório.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logging_config import configure_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

_SINTOMAS = {
    "normal": [
        "paciente refere leve desconforto abdominal, sem outros sintomas",
        "queixa de dor de cabeça leve, nega febre ou vômitos",
        "consulta de rotina, exames laboratoriais dentro da normalidade",
        "leve congestão nasal e espirros, sem febre",
        "dor lombar leve após esforço físico, sem irradiação",
        "revisão de rotina, paciente assintomático",
        "pequeno corte superficial no dedo, sem sinais de infecção",
        "queixa de cansaço leve após atividade física intensa",
        "coceira leve na pele, sem lesões visíveis",
        "consulta de acompanhamento, pressão arterial normal",
    ],
    "atencao": [
        "febre moderada (38.2C) há 2 dias, associada a tosse persistente",
        "dor abdominal moderada, com náuseas intermitentes",
        "paciente hipertenso com pressão arterial elevada (150/95), assintomático",
        "dor torácica leve ao esforço, sem histórico cardíaco relevante",
        "vômitos recorrentes há 24 horas, sinais leves de desidratação",
        "ferimento com sangramento controlado, necessita sutura",
        "falta de ar leve a moderada em paciente asmático conhecido",
        "diabético com glicemia alterada (220 mg/dL), sem outros sintomas agudos",
        "dor lombar intensa com limitação de movimento",
        "quadro de gastroenterite com diarreia há 2 dias",
    ],
    "urgente": [
        "dor torácica intensa e súbita, irradiando para o braço esquerdo",
        "dificuldade respiratória grave, saturação de oxigênio abaixo de 90%",
        "paciente inconsciente, sem resposta a estímulos verbais",
        "sangramento ativo intenso após trauma, sinais de choque hipovolêmico",
        "convulsão em curso, histórico de epilepsia não controlada",
        "febre alta (40C) com rigidez de nuca e confusão mental",
        "suspeita de AVC: fraqueza súbita em um lado do corpo, fala arrastada",
        "dor abdominal súbita e intensa em quadrante inferior direito, "
        "suspeita de apendicite aguda",
        "reação alérgica grave com edema de glote e dificuldade para respirar",
        "trauma cranioencefálico após queda de altura, paciente confuso",
    ],
}

_PREFIXOS = [
    "Paciente do sexo {sexo}, {idade} anos, comparece à emergência.",
    "Laudo de atendimento: paciente {sexo}, {idade} anos.",
    "Registro de triagem — paciente {idade} anos, sexo {sexo}.",
    "Avaliação inicial: paciente {sexo}, {idade} anos, admitido(a) na unidade.",
]

_SEXOS = ["masculino", "feminino"]


def _gerar_texto(label: str, rng: random.Random) -> str:
    prefixo = rng.choice(_PREFIXOS).format(
        sexo=rng.choice(_SEXOS), idade=rng.randint(1, 95)
    )
    sintoma = rng.choice(_SINTOMAS[label])
    return f"{prefixo} Relato clínico: {sintoma}."


def generate_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Gera `n_samples` laudos sintéticos balanceados entre as 3 classes."""
    rng = random.Random(seed)
    labels = list(_SINTOMAS.keys())
    rows = []
    for i in range(n_samples):
        label = labels[i % len(labels)]
        rows.append({"text": _gerar_texto(label, rng), "label": label})
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    """Gera o CSV sintético em data/raw/ conforme config.yaml."""
    configure_logging()
    cfg = load_config()
    set_seed(cfg["split"]["random_state"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples", type=int, default=cfg["data"]["n_synthetic_samples"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(cfg["data"]["raw_path"]) / cfg["data"]["raw_file"],
    )
    args = parser.parse_args()

    df = generate_dataset(args.n_samples, cfg["split"]["random_state"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    logger.info(
        "Dataset sintético gerado em %s (%d amostras, distribuição: %s)",
        args.output,
        len(df),
        df["label"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
