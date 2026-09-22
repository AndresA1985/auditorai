"""Exercise the real wrapper/schema without DB, model loading or credentials."""
import ast
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.schemas import PrediccionRequest, PrediccionResponse
from app.tariff_knowledge import annotate_excel_prediction
from app.features import build_prediction_text
from test_tariff_pdf_contract import isolated_app


CPRE="CPRE. Se canula la vía biliar bajo guía fluoroscópica. Se realiza esfinterotomía. Se coloca una prótesis biliar."


def wrapper(original):
    path=Path(__file__).parents[1]/"app"/"predictor.py"
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="predecir"]
    namespace={"PrediccionRequest":PrediccionRequest,"Tuple":Tuple,
               "annotate_excel_prediction":annotate_excel_prediction,"_predecir_original":original}
    exec(compile(tree,str(path),"exec"),namespace)
    return namespace["predecir"]


def test_http_retains_excel_lines_and_original_prediction():
    original={"codigos":["99999"],"honorarios_codigo":{"99999":"75"},"honorario":"99999(75)",
              "tiempo_anestesia":99,"observacion_auditor":"Comparar."}
    before=deepcopy(original)
    predict=wrapper(lambda _: (original,0.2))
    response=TestClient(isolated_app(predict)).post("/predecir_auditoria",json={"id_agenda":1,"hallazgos_conclusion":CPRE})
    assert response.status_code==200
    data=response.json()["prediccion"]
    assert all(data[k]==v for k,v in before.items())
    excel=data["propuesta_planillaje_excel"]
    assert excel["estado"]=="propuesta"
    assert [(l["codigo"],l["porcentaje"]) for l in excel["lineas"]]==[("43273",100),("43262",50),("43268",100),("240159",100)]
    assert excel["tiempo_anestesia"]==4
    assert excel["principal"]["codigo"] is None
    assert original==before


def test_excel_remains_available_when_history_has_no_candidate():
    def absent(_): raise ValueError("No existe historico codificado para esos filtros.")
    result,score=wrapper(absent)(PrediccionRequest(id_agenda=1,hallazgos_conclusion=CPRE))
    assert result["codigos"]==[] and score==0
    assert result["propuesta_planillaje_excel"]["estado"]=="propuesta"
    PrediccionResponse(mensaje="Revisar",prediccion=result)
    def broken(_): raise ValueError("Otro error interno")
    with pytest.raises(ValueError,match="Otro error"): wrapper(broken)(PrediccionRequest(id_agenda=1))


def test_billing_metadata_never_enters_features_and_duplicate_reports_rejected():
    fields={"id_agenda":1,"hallazgos_conclusion":"Vía oral. Se toma biopsia de antro."}
    base=PrediccionRequest(**fields)
    with_reports=PrediccionRequest(**fields,modalidad_planillaje="abierto",nivel_sala="crm_segundo",
        informes_013b=[{"id_informe":"INFORME_SINTETICO","sitios":["antro"]}])
    assert build_prediction_text(base)==build_prediction_text(with_reports)
    with pytest.raises(ValidationError):
        PrediccionRequest(**fields,informes_013b=[{"id_informe":"x","sitios":["antro"]},{"id_informe":" x ","sitios":["antro"]}])
