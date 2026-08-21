import html
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Tuple
from .anestesia_model import aplicar_tiempos_anestesia
from .config import settings
from .db import fetch_all
from .honorarios_model import aplicar_honorarios
from .ml_model import MODEL_PATH, predecir_con_modelo
from .schemas import PrediccionRequest
from .template_reference import anexar_soporte_plantillas

STOPWORDS = {
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "el", "en",
    "entre", "es", "esta", "las", "la", "lo", "los", "para", "por", "que", "se", "sin",
    "su", "sus", "un", "una", "y", "o", "u", "via", "oral", "rectal", "normal"
}


def limpiar_texto(valor: str) -> str:
    if not valor:
        return ""
    texto = html.unescape(str(valor))
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokens(valor: str) -> Counter:
    return Counter(t for t in limpiar_texto(valor).split() if len(t) > 2 and t not in STOPWORDS)


def jaccard_ponderado(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    comunes = sum((a & b).values())
    total = sum((a | b).values())
    return comunes / total if total else 0.0


def separar_codigos(valor: str) -> List[str]:
    if not valor:
        return []
    partes = re.split(r"[,+]", str(valor))
    codigos = []
    for parte in partes:
        codigo = parte.strip()
        if codigo and codigo not in codigos:
            codigos.append(codigo)
    return codigos


def parse_honorarios(valor: str, codigos: Iterable[str]) -> Dict[str, str]:
    honorarios: Dict[str, str] = {}
    if valor:
        for codigo, porcentaje in re.findall(r"([^(),+]+)\(([^()]*)\)", str(valor)):
            codigo = codigo.strip()
            porcentaje = porcentaje.strip()
            if codigo and porcentaje:
                honorarios[codigo] = porcentaje
    for codigo in codigos:
        honorarios.setdefault(codigo, "100")
    return honorarios


def buscar_grupos_historicos(req: PrediccionRequest) -> List[dict]:
    sql = """
        SELECT
            id_agenda,
            id_empresa,
            id_seguro,
            codigo_grupo_auditor,
            honorario_auditor,
            tiempo_anestesia,
            nombre_procedimiento,
            observacion_auditor,
            procedimiento_auditor,
            GROUP_CONCAT(DISTINCT codigo ORDER BY codigo SEPARATOR ',') AS codigos,
            MAX(hallazgo) AS hallazgo,
            MAX(conclusion) AS conclusion,
            COUNT(*) AS detalles
        FROM ap_auditoria_doctor_detalle
        WHERE estado = 1
          AND codigo IS NOT NULL
          AND codigo <> ''
          AND id_agenda <> %(id_agenda)s
          AND (%(id_empresa)s IS NULL OR id_empresa = %(id_empresa)s)
          AND (%(id_seguro)s IS NULL OR id_seguro = %(id_seguro)s)
        GROUP BY
            id_agenda,
            id_empresa,
            id_seguro,
            codigo_grupo_auditor,
            honorario_auditor,
            tiempo_anestesia,
            nombre_procedimiento,
            observacion_auditor,
            procedimiento_auditor
        ORDER BY MAX(updated_at) DESC
        LIMIT 5000
    """
    return fetch_all(sql, {
        "id_agenda": req.id_agenda,
        "id_empresa": req.id_empresa or None,
        "id_seguro": req.id_seguro or None,
    })


def texto_request(req: PrediccionRequest) -> str:
    return " ".join([
        req.procedimiento_sistema or "",
        req.hallazgos_conclusion or "",
        req.descripcion_estudio_013 or "",
    ])


def texto_grupo(row: dict) -> str:
    return " ".join([
        str(row.get("nombre_procedimiento") or ""),
        str(row.get("procedimiento_auditor") or ""),
        str(row.get("hallazgo") or ""),
        str(row.get("conclusion") or ""),
    ])


def puntuar(req: PrediccionRequest, row: dict, entrada_tokens: Counter) -> float:
    score = jaccard_ponderado(entrada_tokens, tokens(texto_grupo(row)))
    proc_req = limpiar_texto(req.procedimiento_sistema or "")
    proc_hist = limpiar_texto(" ".join([str(row.get("nombre_procedimiento") or ""), str(row.get("procedimiento_auditor") or "")]))
    if proc_req and proc_hist and proc_req in proc_hist:
        score += settings.score_exact_match_bonus
    elif proc_req and proc_hist and any(tok in proc_hist for tok in proc_req.split() if len(tok) > 4):
        score += settings.score_token_match_bonus
    if row.get("codigo_grupo_auditor"):
        score += settings.score_code_group_bonus
    return score


def predecir(req: PrediccionRequest) -> Tuple[dict, float]:
    if MODEL_PATH.exists():
        prediccion, score = predecir_con_modelo(req)
        prediccion, score = anexar_soporte_plantillas(texto_request(req), prediccion, score)
        prediccion = aplicar_honorarios(req, prediccion)
        return aplicar_tiempos_anestesia(req, prediccion), score

    entrada = tokens(texto_request(req))
    grupos = buscar_grupos_historicos(req)

    if not grupos:
        raise ValueError("No existe historico codificado para esos filtros.")

    candidatos = []
    for row in grupos:
        score = puntuar(req, row, entrada)
        candidatos.append((score, row))

    candidatos.sort(key=lambda item: item[0], reverse=True)
    score, mejor = candidatos[0]

    if score < settings.min_score:
        raise ValueError("No se encontro una referencia historica suficientemente parecida.")

    codigos = separar_codigos(mejor.get("codigo_grupo_auditor") or mejor.get("codigos") or "")
    honorarios_codigo = parse_honorarios(mejor.get("honorario_auditor") or "", codigos)
    honorario = ",".join("{0}({1})".format(codigo, honorarios_codigo[codigo]) for codigo in codigos)

    return {
        "codigos": codigos,
        "honorarios_codigo": honorarios_codigo,
        "honorario": honorario,
        "tiempo_anestesia": mejor.get("tiempo_anestesia") or "",
        "nombre_procedimiento": mejor.get("nombre_procedimiento") or mejor.get("procedimiento_auditor") or req.procedimiento_sistema or "",
        "observacion_auditor": "Propuesta generada por IA; validar antes de guardar.",
    }, score


def clases_codigos() -> List[dict]:
    sql = """
        SELECT
            codigo,
            COUNT(*) AS frecuencia,
            COUNT(DISTINCT id_agenda) AS agendas,
            GROUP_CONCAT(DISTINCT porcentaje ORDER BY porcentaje SEPARATOR ' | ') AS porcentajes_usados,
            GROUP_CONCAT(DISTINCT tiempo_anestesia ORDER BY tiempo_anestesia SEPARATOR ' | ') AS tiempos_anestesia_usados
        FROM ap_auditoria_doctor_detalle
        WHERE estado = 1
          AND codigo IS NOT NULL
          AND codigo <> ''
        GROUP BY codigo
        ORDER BY frecuencia DESC, codigo ASC
    """
    rows = fetch_all(sql)
    for row in rows:
        frecuencia = int(row.get("frecuencia") or 0)
        if frecuencia >= 1000:
            clase = "ALTA_FRECUENCIA"
        elif frecuencia >= 100:
            clase = "MEDIA_FRECUENCIA"
        elif frecuencia >= 10:
            clase = "BAJA_FRECUENCIA"
        else:
            clase = "RARA_REVISAR"
        row["clase_frecuencia"] = clase
    return rows