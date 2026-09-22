"""Source fidelity and billing contract tests; not clinical performance metrics."""
from copy import deepcopy
import pytest
from app.tariff_knowledge import propose_billing, load_knowledge, annotate_excel_prediction
from scripts.build_tariff_knowledge import parse_codes, parse_percentages, select_sheets


CPRE = ("CPRE. Se canula la vía biliar bajo guía fluoroscópica. "
        "Se realiza esfinterotomía. Se coloca una prótesis biliar.")
POLYP = ("Vía oral. Se realiza polipectomía con pinza de biopsia. "
         "Se toman biopsias de antro y cuerpo.")


@pytest.mark.parametrize("text,codes,fees,time", [
    (CPRE,["43273","43262","43268","240159"],[100,50,100,100],4),
    (POLYP,["43250","43239"],[100,50],2),
])
def test_source_examples_preserve_pair_order(text,codes,fees,time):
    result=propose_billing({"hallazgos_conclusion":text})
    assert result["estado"]=="propuesta",result["motivos"]
    assert [line["codigo"] for line in result["lineas"]]==codes
    assert [line["porcentaje"] for line in result["lineas"]]==fees
    assert result["tiempo_anestesia"]==time
    assert result["principal"]["codigo"] is None
    assert all(line["fuentes"] for line in result["lineas"])


def test_compiled_source_coverage_and_unrepaired_anomalies():
    data=load_knowledge()
    records={r["fila"]:r for r in data["registros"]}
    assert set(records)==set(range(2,286))
    assert sum(bool(r["codigos"]) for r in records.values())==198
    for row in (112,124,263):
        assert "cardinalidad_honorarios" in records[row]["incidencias"]
    assert records[216]["porcentajes"]==[100,50,100,10]
    assert records[188]["porcentajes"]!=records[204]["porcentajes"]
    assert all(r["tiempo_anestesia"]==2 for r in records.values() if r["alternativa"])


def test_importer_does_not_concatenate_or_fill():
    assert parse_codes("43250 - 43239")==(["43250","43239"],False)
    assert parse_codes("70200003 O 43239 (DEPENDIENDO DEL CV)")==(["70200003","43239"],True)
    assert parse_percentages(1,"0%")==[100]
    assert parse_percentages("100%-50%-100%-10%")==[100,50,100,10]
    assert parse_percentages(None)==[]
    assert parse_percentages("100%")==[100]
    with pytest.raises(ValueError): parse_codes("43250 basura 43239")
    with pytest.raises(ValueError): parse_percentages("100% + falta")


def test_sheet_selection_rejects_trim_collisions():
    class Book:
        sheetnames=["hallazgos_base","hallazgos_base ","HONORARIOS "]
    with pytest.raises(ValueError,match="ambigua"): select_sheets(Book())


def test_model_output_is_preserved_and_old_proposal_is_recomputed():
    original={"codigos":["99999"],"honorarios_codigo":{"99999":"75"},
              "tiempo_anestesia":99,"codigo_scores":{"99999":0.2}}
    before=deepcopy(original)
    updated=annotate_excel_prediction(original,{"hallazgos_conclusion":CPRE})
    assert original==before
    assert all(updated[key]==value for key,value in before.items())
    updated=annotate_excel_prediction(updated,{"hallazgos_conclusion":"Plan: CPRE"})
    assert updated["propuesta_planillaje_excel"]["estado"]=="abstencion"
    assert updated["propuesta_planillaje_excel"]["lineas"]==[]


@pytest.mark.parametrize("finding",[
    "Vía oral. Se toma biopsia de antro dentro de dos días.",
    "Vía oral. Se toma biopsia de antro con intención de realizarla.",
    "Vía oral. Se toma biopsia de antro y se canceló.",
    "Vía oral. Se toma biopsia de antro como ejemplo.",
    "Vía oral. Si hay lesiones se toma biopsia de antro.",
    "Vía oral. Si es normal se toma biopsia de antro.",
    "Vía oral. No consta que se toma biopsia de antro.",
    "Vía oral. No se confirma que se toma biopsia de antro.",
    "Vía oral. Se decide colocar 3 ligas.",
    "Vía oral. Se realiza ligadura de varices. No se hizo ligadura.",
    "CPRE. Se canula vía biliar. Se realiza esfinterotomía. Se intentará colocar stent. Bajo guía fluoroscópica.",
    "CPRE. Se canula vía biliar. Se realiza esfinterotomía. Se decide colocar stent. Bajo guía fluoroscópica.",
    "No se realizó CPRE. Se canula vía biliar. Se realiza esfinterotomía. Se coloca stent. Bajo guía fluoroscópica.",
])
def test_reviewed_adversarial_assertions_do_not_bill(finding):
    result=propose_billing({"hallazgos_conclusion":finding},
        {"estado":"extraido","codigos":[{"codigo":"43239"}],"advertencias":[]})
    assert result["estado"]=="abstencion"
    assert result["lineas"]==[]


@pytest.mark.parametrize("site,code",[("papila","280017"),("lesión de papila de Vater","280025"),
                                    ("lesión tumoral","280025"),("hígado","280095")])
def test_biopsy_site_directives_keep_compound_anatomy(site,code):
    result=propose_billing({"hallazgos_conclusion":"CPRE. Se toma biopsia de "+site+".",
        "informes_013b":[{"id_informe":"synthetic-report","sitios":[site]}]})
    lines=result["complementos"]["biopsias"]["lineas"]
    assert [(line["codigo"],line["cantidad"],line["porcentaje"]) for line in lines]==[(code,1,None)]
