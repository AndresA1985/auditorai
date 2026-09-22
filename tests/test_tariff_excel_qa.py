"""QA contract for Excel-grounded, additive billing proposals.

The fixtures are synthetic and contain no patient or workbook clinical text.
"""
from copy import deepcopy
import json

import pytest

from app import tariff_knowledge as knowledge


UPPER = (
    "ENDOSCOPIA DIGESTIVA ALTA. VIA ORAL. "
    "SE TOMA BIOPSIA DE ANTRO Y CUERPO."
)
LOWER = (
    "COLONOSCOPIA. VIA ANAL. SE INTRODUCE COLONOSCOPIO HASTA CIEGO. "
    "SE TOMA BIOPSIA DE COLON ASCENDENTE."
)


def evidence(*codes, warnings=(), state="extraido"):
    return {
        "estado": state,
        "fuente": "texto" if codes else "ninguna",
        "codigos": [{"codigo": code} for code in codes],
        "advertencias": list(warnings),
        "documento_sha256": "a" * 64 if codes else None,
    }


def test_compiled_knowledge_is_pinned_complete_and_deidentified():
    compiled = knowledge.load_knowledge()
    assert compiled["fuente_sha256"] == knowledge.SOURCE_SHA256
    assert compiled["hojas"] == {
        "precedentes": "hallazgos_base",
        "directrices": "HONORARIOS ",
    }

    pairs = [tuple(record["codigos"]) for record in compiled["registros"]]
    assert pairs.count(("70200003", "43239")) == 55
    assert pairs.count(("70200004", "45380")) == 24

    serialized = json.dumps(compiled, ensure_ascii=False).lower()
    for private_field in (
        "cedula",
        "paciente",
        "fecha_nacimiento",
        "id_hc",
        "hallazgos_conclusion",
        "descripcion_estudio_013",
    ):
        assert private_field not in serialized


@pytest.mark.parametrize(
    "finding,document_codes,expected_code,expected_alternatives",
    [
        (UPPER, ("70200003",), "70200003", {"70200003", "43239"}),
        (UPPER, ("43239",), "43239", {"70200003", "43239"}),
        (LOWER, ("70200004",), "70200004", {"70200004", "45380"}),
        (LOWER, ("45380",), "45380", {"70200004", "45380"}),
        (UPPER, ("70200003", "70200004"), "70200003", {"70200003", "43239"}),
        (LOWER, ("70200003", "70200004"), "70200004", {"70200004", "45380"}),
    ],
)
def test_requested_alternatives_keep_code_fee_pair_and_combo_time(
    finding, document_codes, expected_code, expected_alternatives
):
    result = knowledge.propose_billing(
        {"hallazgos_conclusion": finding}, evidence(*document_codes)
    )

    assert result["estado"] == "propuesta"
    assert result["motivos"] == []
    assert result["tiempo_anestesia"] == 2
    assert [(line["orden"], line["codigo"], line["porcentaje"])
            for line in result["lineas"]] == [(1, expected_code, 100)]
    assert result["lineas"][0]["rol"] == "por_confirmar"
    assert result["principal"]["codigo"] is None
    assert result["decision_final"] == "auditor"

    candidate_codes = {
        candidate["lineas"][0]["codigo"] for candidate in result["candidatos"]
    }
    assert candidate_codes == expected_alternatives
    sources = result["lineas"][0]["fuentes"]
    assert sources
    assert all(source["celdas"]["codigos"] == f"AA{source['fila']}" for source in sources)
    assert all(source["celdas"]["porcentajes"] == f"AB{source['fila']}" for source in sources)
    assert all(source["celdas"]["anestesia"] == f"AC{source['fila']}" for source in sources)
    assert {source["hoja"] for source in sources} == {"hallazgos_base"}


@pytest.mark.parametrize(
    "finding",
    [
        "ENDOSCOPIA DIGESTIVA ALTA. NO SE TOMA BIOPSIA DE ANTRO.",
        "ENDOSCOPIA DIGESTIVA ALTA. SE PLANEA TOMAR BIOPSIA DE ANTRO.",
        "ANTECEDENTES: ENDOSCOPIA ALTA. SE TOMO BIOPSIA DE ANTRO.",
        "VIA ORAL. SE TOMA BIOPSIA DE ANTRO SIN REALIZAR.",
        "VIA ORAL. SI HAY LESION SE TOMA BIOPSIA DE ANTRO.",
        "VIA ORAL. SE TOMA BIOPSIA DE ANTRO. BIOPSIA NO FUE REALIZADA.",
    ],
)
def test_negated_planned_historical_or_unperformed_biopsy_abstains(finding):
    result = knowledge.propose_billing(
        {"hallazgos_conclusion": finding}, evidence("70200003")
    )
    assert result["estado"] == "abstencion"
    assert result["lineas"] == []
    assert result["tiempo_anestesia"] is None
    assert result["motivos"]


def test_requested_name_and_pdf_never_replace_performed_clinical_evidence():
    result = knowledge.propose_billing(
        {
            "procedimiento_sistema": UPPER,
            "hallazgos_conclusion": "SIN HALLAZGOS DE PROCEDIMIENTO REALIZADO.",
        },
        evidence("70200003"),
    )
    assert result["estado"] == "abstencion"
    assert result["lineas"] == []
    assert "sin_actos_realizados_suficientes" in result["motivos"]


def test_conflicting_upper_and_lower_performed_events_abstain():
    result = knowledge.propose_billing(
        {"hallazgos_conclusion": UPPER + " " + LOWER},
        evidence("70200003", "70200004"),
    )
    assert result["estado"] == "abstencion"
    assert result["lineas"] == []
    assert result["motivos"]


@pytest.mark.parametrize(
    "document",
    [
        None,
        evidence("70200003", "43239"),
        evidence("70200003", warnings=("tabla_ambigua",)),
    ],
)
def test_missing_ambiguous_or_unreliable_document_cannot_choose_an_alternative(document):
    result = knowledge.propose_billing(
        {"hallazgos_conclusion": UPPER}, document
    )
    assert result["estado"] == "abstencion"
    assert result["lineas"] == []
    assert result["tiempo_anestesia"] is None
    assert "alternativa_requiere_codigo_documental_inequivoco" in result["motivos"]


def test_annotation_is_additive_and_preserves_every_original_model_field():
    original = {
        "codigos": ["43235"],
        "codigo_scores": {"43235": 0.71},
        "codigo_ranking": [{"codigo": "43235", "score": 0.71, "selected": True}],
        "honorarios_codigo": {"43235": "75"},
        "honorario": "43235(75)",
        "tiempos_anestesia_codigo": {"43235": 4},
        "tiempo_anestesia": 4,
        "nombre_procedimiento": "Procedimiento sintetico QA",
        "propuesta_tarifario_documental": {"sentinel": "preservar"},
        "requiere_revision": False,
    }
    before = deepcopy(original)

    annotated = knowledge.annotate_excel_prediction(
        original, {"hallazgos_conclusion": UPPER}, evidence("70200003")
    )

    assert original == before
    for field, value in before.items():
        assert annotated[field] == value
    assert annotated["propuesta_planillaje_excel"]["estado"] == "propuesta"
    assert annotated["propuesta_planillaje_excel"]["requiere_revision"] is True
    assert annotated["propuesta_planillaje_excel"]["lineas"][0]["codigo"] == "70200003"


def test_013b_counts_reports_not_fragments_and_open_room_is_reviewable():
    result = knowledge.propose_billing(
        {
            "hallazgos_conclusion": LOWER,
            "modalidad_planillaje": "abierto",
            "nivel_sala": "crm_segundo",
            "informes_013b": [
                {"id_informe": "R-1", "sitios": ["colon ascendente"]},
                {"id_informe": "R-2", "sitios": ["colon ascendente", "colon ascendente"]},
            ],
        },
        evidence("45380"),
    )

    assert result["estado"] == "propuesta"
    biopsy = result["complementos"]["biopsias"]
    assert biopsy["estado"] == "requiere_revision"
    assert [(line["codigo"], line["cantidad"], line["porcentaje"])
            for line in biopsy["lineas"]] == [("280009", 2, None)]
    assert biopsy["lineas"][0]["motivos"] == ["porcentaje_no_especificado"]
    assert biopsy["lineas"][0]["fuentes"] == [
        {"hoja": "HONORARIOS ", "celdas": ["B14", "A13"]}
    ]

    room = result["complementos"]["sala"]
    assert room["estado"] == "requiere_revision"
    assert [(item["codigo"], item["porcentaje"]) for item in room["opciones"]] == [
        ("395162", 100),
        ("395272", 100),
    ]


def test_invalid_013b_does_not_invent_count_and_gastro_room_keeps_its_pair():
    result = knowledge.propose_billing(
        {
            "hallazgos_conclusion": LOWER,
            "modalidad_planillaje": "abierto",
            "nivel_sala": "gastro_tercero",
            "informes_013b": [
                {"id_informe": "R-1", "sitios": ["colon"]},
                {"id_informe": "R-1", "sitios": ["colon"]},
            ],
        },
        evidence("45380"),
    )

    biopsy_line = result["complementos"]["biopsias"]["lineas"][0]
    assert biopsy_line["cantidad"] is None
    assert "informes_013b_no_verificados" in biopsy_line["motivos"]
    assert [item["codigo"] for item in result["complementos"]["sala"]["opciones"]] == [
        "395173",
        "395281",
    ]


def test_package_codes_do_not_add_biopsy_or_room_and_modality_conflict_abstains():
    package = knowledge.propose_billing(
        {"hallazgos_conclusion": UPPER}, evidence("70200003")
    )
    assert package["estado"] == "propuesta"
    for complement in package["complementos"].values():
        assert complement["estado"] == "no_agregado"
        assert complement["motivo"] == "paquete_sin_adicionales"

    conflict = knowledge.propose_billing(
        {"hallazgos_conclusion": UPPER, "modalidad_planillaje": "abierto"},
        evidence("70200003"),
    )
    assert conflict["estado"] == "abstencion"
    assert conflict["lineas"] == []
    assert conflict["motivos"] == ["modalidad_contradice_codigo"]


def _synthetic_record(*, fees, anesthesia, issues=(), row=900):
    return {
        "fila": row,
        "codigos": ["43250", "43239"],
        "alternativa": False,
        "porcentajes": list(fees),
        "tiempo_anestesia": anesthesia,
        "caracteristicas": {
            "familias": ["alta"],
            "actos": ["biopsia_alta"],
            "sitios_biopsia": ["antro"],
            "incertidumbres": [],
        },
        "incidencias": list(issues),
        "celdas": {
            "clinica": [f"Y{row}", f"Z{row}"],
            "contexto": [f"Q{row}", f"AD{row}"],
            "codigos": f"AA{row}",
            "porcentajes": f"AB{row}",
            "anestesia": f"AC{row}",
            "notas": f"AE{row}",
        },
    }


def test_ordered_code_percentage_pairs_are_never_reassigned_by_position():
    rules = knowledge.load_knowledge()["reglas"]
    candidate = knowledge.candidate(
        [_synthetic_record(fees=(100, 50), anesthesia=2)], rules
    )
    assert [(line["orden"], line["codigo"], line["porcentaje"])
            for line in candidate["lineas"]] == [
        (1, "43250", 100),
        (2, "43239", 50),
    ]
    assert candidate["tiempo_anestesia"] == 2


@pytest.mark.parametrize(
    "records,expected_reason",
    [
        (
            [
                _synthetic_record(fees=(100, 50), anesthesia=2, row=900),
                _synthetic_record(fees=(50, 100), anesthesia=2, row=901),
            ],
            "precedentes_discrepantes",
        ),
        (
            [_synthetic_record(
                fees=(100, 50),
                anesthesia=None,
                issues=("anestesia_ausente",),
            )],
            "anestesia_ausente",
        ),
        (
            [_synthetic_record(
                fees=(100,),
                anesthesia=2,
                issues=("cardinalidad_honorarios",),
            )],
            "cardinalidad_honorarios",
        ),
    ],
)
def test_conflicting_fees_missing_anesthesia_or_unpaired_fee_abstains(
    monkeypatch, records, expected_reason
):
    compiled = deepcopy(knowledge.load_knowledge())
    compiled["registros"] = records
    monkeypatch.setattr(knowledge, "load_knowledge", lambda: compiled)

    result = knowledge.propose_billing({"hallazgos_conclusion": UPPER})

    assert result["estado"] == "abstencion"
    assert result["lineas"] == []
    assert result["tiempo_anestesia"] is None
    assert expected_reason in result["motivos"]
    assert result["principal"]["codigo"] is None


def test_unavailable_compiled_knowledge_fails_closed(monkeypatch):
    def unavailable():
        raise ValueError("synthetic invalid knowledge")

    monkeypatch.setattr(knowledge, "load_knowledge", unavailable)
    result = knowledge.propose_billing({"hallazgos_conclusion": UPPER})
    assert result["estado"] == "abstencion"
    assert result["lineas"] == []
    assert result["motivos"] == ["conocimiento_no_disponible"]
