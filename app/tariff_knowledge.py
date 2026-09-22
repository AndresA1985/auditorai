"""Auditable Excel-only billing proposals, independent of model predictions."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from .tariff_semantics import VERSION, clinical_features, site_labels

SOURCE_SHA256 = "e6766c3acbda7750515158eca1edaa9446b5aca3ea2f42b0b41bb0b79caf13fa"
KNOWLEDGE_PATH = Path(__file__).with_name("data") / "tariff_knowledge.json"
NON_BLOCKING_WARNINGS = {"codigo_validacion_no_verificable", "codigo_validacion_requiere_revision", "requiere_verificacion_ocr"}


@lru_cache(maxsize=1)
def load_knowledge():
    data = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    if (data.get("schema_version") != "1" or data.get("semantics_version") != VERSION
            or data.get("fuente_sha256") != SOURCE_SHA256
            or data.get("hojas") != {"precedentes":"hallazgos_base", "directrices":"HONORARIOS "}):
        raise ValueError("Base de conocimiento no verificada")
    return data


def source_ref(record):
    return {"hoja":"hallazgos_base", "fila":record["fila"], "celdas":deepcopy(record["celdas"])}


def directive(cells):
    return {"hoja":"HONORARIOS ", "celdas":cells}


def biopsy_alternative(code):
    """Values and source rows are read from the compiled worksheet, never ML."""
    records=[r for r in load_knowledge()["registros"] if r["alternativa"] and code in r["codigos"]]
    values={(tuple(r["porcentajes"]),r["tiempo_anestesia"]) for r in records}
    if len(values)!=1: raise ValueError("Valores alternativos contradictorios")
    percentages,time=next(iter(values))
    if len(percentages)!=1 or time is None: raise ValueError("Alternativa incompleta")
    return {"porcentaje":percentages[0],"tiempo_anestesia":time,
            "fuente_sha256":SOURCE_SHA256,"fuentes":[source_ref(r) for r in records]}


def candidate(records, rules, selected=None):
    first=records[0]
    codes=[selected] if selected else first["codigos"]
    fees=first["porcentajes"]
    aligned=len(fees)==len(codes)
    sources=[source_ref(r) for r in records]
    lines=[]
    for i,code in enumerate(codes):
        role="adicional" if code in rules["adicionales_100"]["codigos"] else "por_confirmar"
        lines.append({"orden":i+1,"codigo":code,"porcentaje":fees[i] if aligned else None,
                      "rol":role,"fuentes":sources})
    issues=sorted({issue for r in records for issue in r["incidencias"]
                   if not issue.startswith("evidencia_semantica")})
    for line in lines:
        if line["rol"]=="adicional" and line["porcentaje"] not in (None,rules["adicionales_100"]["porcentaje"]):
            issues.append("conflicto_directriz_adicional_100")
    return {"lineas":lines,"tiempo_anestesia":first["tiempo_anestesia"],
            "porcentajes_fuente":deepcopy(fees),"incidencias":sorted(set(issues)),"fuentes":sources}


def complements(payload, features, rules, package, unresolved_mode=False):
    result={"biopsias":{"estado":"sin_evidencia","lineas":[],"no_sumar_automaticamente":True},
            "sala":{"estado":"sin_evidencia","opciones":[]}}
    if package or unresolved_mode:
        reason="paquete_sin_adicionales" if package else "modalidad_pendiente"
        return {name:{**item,"estado":"no_agregado","motivo":reason} for name,item in result.items()}
    supported=set(features["sitios_biopsia"])
    if features["incertidumbres"]: supported=set()
    options=[r for r in rules["biopsias"] if supported & set(r["sitios"])]
    counts=defaultdict(int)
    reports=payload.get("informes_013b")
    report_issue=False
    if reports:
        seen=set()
        for report in reports:
            if (not isinstance(report,dict) or not isinstance(report.get("id_informe"),str)
                    or not report["id_informe"].strip() or report["id_informe"].strip() in seen
                    or not isinstance(report.get("sitios"),list) or not report["sitios"]
                    or any(not isinstance(site,str) or not site.strip() for site in report["sitios"])):
                report_issue=True
                continue
            seen.add(report["id_informe"].strip())
            sites=set().union(*(site_labels(site) for site in report["sitios"]))
            mapped={r["codigo"] for r in options if sites & set(r["sitios"])}
            if not sites or not sites <= supported or len(mapped)!=1:
                report_issue=True
            else: counts[next(iter(mapped))]+=1
    for rule in options:
        result["biopsias"]["lineas"].append({"codigo":rule["codigo"],"porcentaje":None,
            "cantidad":counts[rule["codigo"]] if reports and not report_issue and counts[rule["codigo"]] else None,
            "fuentes":[directive([rule["celda"],"A13"])],
            "motivos":["porcentaje_no_especificado"] + ([] if reports and not report_issue and counts[rule["codigo"]] else ["informes_013b_no_verificados"])})
    if options: result["biopsias"]["estado"]="requiere_revision"
    mode=payload.get("modalidad_planillaje")
    level=payload.get("nivel_sala")
    if mode=="abierto":
        if level in ("crm_segundo","gastro_tercero"):
            result["sala"]={"estado":"requiere_revision","motivo":"seleccionar_derechos_aplicables",
                "opciones":[{"codigo":c,"porcentaje":rules["sala"]["porcentaje"],
                             "fuentes":[directive(rules["sala"]["celdas"])]} for c in rules["sala"][level]]}
        else: result["sala"].update(estado="abstencion",motivo="nivel_sala_no_indicado")
    return result


def propose_billing(payload, evidence=None):
    if hasattr(payload,"model_dump"): payload=payload.model_dump()
    payload=dict(payload)
    if evidence is None: evidence=payload.get("evidencia_tarifario") or {}
    if hasattr(evidence,"model_dump"): evidence=evidence.model_dump()
    output={"schema_version":"1","fuente_sha256":SOURCE_SHA256,
            "estado":"abstencion","requiere_revision":True,"decision_final":"auditor",
            "lineas":[],"tiempo_anestesia":None,"candidatos":[],"motivos":[],
            "fuentes_permitidas":["hallazgos_base","HONORARIOS "]}
    try: knowledge=load_knowledge()
    except (OSError,ValueError,KeyError):
        output["motivos"]=["conocimiento_no_disponible"]
        return output
    features=clinical_features(payload)
    output["evidencia_clinica"]=features
    rules=knowledge["reglas"]
    usable=[r for r in knowledge["registros"] if r["codigos"] and
            not any(i.startswith("evidencia_semantica") for i in r["incidencias"])]
    matched=[r for r in usable if r["caracteristicas"]["familias"]==features["familias"]
             and r["caracteristicas"]["actos"]==features["actos"]]
    package=payload.get("modalidad_planillaje")=="paquete"
    unresolved_mode=False
    if features["incertidumbres"]:
        output["motivos"]=["evidencia_clinica_incierta"]+features["incertidumbres"]
    elif len(features["familias"])!=1 or not features["actos"]:
        output["motivos"]=["sin_actos_realizados_suficientes"]
    elif not matched:
        output["motivos"]=["sin_precedente_equivalente"]
    else:
        alternatives=[r for r in matched if r["alternativa"]]
        if alternatives:
            # Reconcile the clinical family first. The PDF can authorize one
            # code, but cannot supply an act or enter model features.
            codes=alternatives[0]["codigos"]
            available={c.get("codigo") for c in evidence.get("codigos",[]) if isinstance(c,dict)}
            approved=set(codes)&available
            reliable=(evidence.get("estado")=="extraido" and not
                      set(evidence.get("advertencias",[]))-NON_BLOCKING_WARNINGS)
            output["candidatos"]=[candidate(alternatives,rules,c) for c in codes]
            if reliable and len(approved)==1:
                code=next(iter(approved))
                selected=candidate(alternatives,rules,code)
                package=code.startswith("702")
                if (payload.get("modalidad_planillaje")=="abierto" and package
                        or payload.get("modalidad_planillaje")=="paquete" and not package):
                    output["motivos"]=["modalidad_contradice_codigo"]
                else:
                    output.update(estado="propuesta",lineas=selected["lineas"],tiempo_anestesia=selected["tiempo_anestesia"])
            else:
                unresolved_mode=True
                output["motivos"]=["alternativa_requiere_codigo_documental_inequivoco"]
        else:
            grouped=defaultdict(list)
            for r in matched:
                grouped[(tuple(r["codigos"]),tuple(r["porcentajes"]),r["tiempo_anestesia"])].append(r)
            candidates=[candidate(records,rules) for records in grouped.values()]
            output["candidatos"]=candidates
            if len(candidates)!=1:
                output["motivos"]=["precedentes_discrepantes"]
            elif candidates[0]["incidencias"]:
                output["motivos"]=candidates[0]["incidencias"]
            elif package:
                output["motivos"]=["modalidad_paquete_sin_precedente_equivalente"]
            else:
                output.update(estado="propuesta",lineas=candidates[0]["lineas"],tiempo_anestesia=candidates[0]["tiempo_anestesia"])
    # When the wording or source is incomplete, expose bounded related
    # precedents for the auditor. They never become billable lines or fill a
    # missing percentage/time. Matching uses acts/anatomy, never name lookup.
    if output["estado"]=="abstencion" and not output["candidatos"] and features["actos"]:
        related=[]
        requested=set(features["actos"])
        for record in knowledge["registros"]:
            source=record["caracteristicas"]
            common=requested & set(source["actos"])
            if record["codigos"] and common and source["familias"]==features["familias"]:
                related.append((len(requested ^ set(source["actos"])),record["fila"],record))
        output["referencias_parciales"]=[]
        for _,_,record in sorted(related,key=lambda entry:entry[:2])[:8]:
            reference=candidate([record],rules)
            reference.update(solo_referencia=True,alternativa=record["alternativa"],
                incidencias=record["incidencias"],
                actos_comunes=sorted(requested & set(record["caracteristicas"]["actos"])),
                actos_no_acreditados=sorted(set(record["caracteristicas"]["actos"])-requested),
                actos_sin_correspondencia=sorted(requested-set(record["caracteristicas"]["actos"])))
            output["referencias_parciales"].append(reference)
    output["complementos"]=complements(payload,features,rules,package,unresolved_mode)
    output["directrices"]=[directive(["A5","A7","A9","B10","A11","B12"])]
    output["principal"]={"codigo":None,"motivo":"la_posicion_y_el_100_no_determinan_principal"}
    return output


def annotate_excel_prediction(prediction, payload, evidence=None):
    result=deepcopy(prediction)
    result["propuesta_planillaje_excel"]=propose_billing(payload,evidence)
    return result
