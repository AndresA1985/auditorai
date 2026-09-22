"""Clinical abstractions, never a tariff catalog or a statistical classifier.

Vocabulary is grounded in hallazgos_base Y/Z/AD. Only performed acts in findings
or the actual report count; the requested procedure name cannot prove an act.
Unknown/qualified interventions are returned for review instead of discarded.
"""
import re
from .tariff_clinical import (
    _normalize_findings, _biopsy_support, _biopsy_denied, _procedure_denied,
    SECTION, PERFORMED_SECTIONS,
)

VERSION = "excel_clinical_acts_v1"
MODAL = re.compile(
    r"\b(?:no|nunca|tampoco|ni|pendiente|plan(?:ea|ifica|ear|ificar)?|"
    r"programad[oa]|recomienda|sugiere|solicita|indica|antecedentes?|"
    r"previ[oa]s?|previamente|anterior(?:es)?|historico|hipotetico|"
    r"si|cuando|manana|posible|podria|debera|debe|necesari[oa]|cancelad[oa]|"
    r"suspendid[oa]|intenta|fallid[oa])\b|\bpor\s+realizar\b|"
    r"\bsin\s+(?:realizar|tomar|efectuar|obtener|exito)\b|"
    r"\ben\s+(?:19|20)[0-9]{2}\b|\bhace\s+.{0,30}\b(?:anos?|meses?|dias?)\b"
)
CONTEXT_MODAL = re.compile(
    r"\b(?:pendiente|plan(?:ea|ifica|ear|ificar)?|programad[oa]|recomienda|sugiere|solicita|indica|"
    r"antecedentes?|previ[oa]s?|previamente|anterior(?:es)?|historico|hipotetico|si|cuando|manana|"
    r"posible|podria|debera|debe|necesari[oa]|cancelad[oa]|suspendid[oa]|intenta|intentara|decide|"
    r"propone|requiere|fallid[oa])\b|\bpor\s+realizar\b|\ben\s+(?:19|20)[0-9]{2}\b|"
    r"\bhace\s+.{0,30}\b(?:anos?|meses?|dias?)\b"
)
DIRECT_DENIAL = re.compile(r"\b(?:no|nunca|tampoco|ni|sin)\s+(?:(?:se|fue|fueron|hizo|hicieron|ha|han|habia|"
    r"pudo|pudieron|logro|lograron|es|sera|posible|realizo|realizaron|realiza|tomar|realizar|efectuo|podra|siempre|debe)\s*){0,6}$")
DO = (r"(?:se\s+)?(?:realizo|realiza(?:mos|n|ndo)?|efectuo|efectua(?:mos)?|"
      r"practico|practica(?:mos)?|procedemos\s+a\s+realizar)\s+(?:una?\s+)?")
ACT_PATTERNS = {
    "canulacion": r"\b(?:se\s+(?:cateteriza|cateterizo|canula|canulo)|cateterizamos|canulamos|procedemos\s+a\s+(?:cateterizar|canular)|canulacion\s+(?:realizada|exitosa))\b.{0,100}\b(?:biliar|coledoco|wirsung)\b",
    "esfinterotomia": DO + r"(?:ampliacion\s+de\s+(?:la\s+)?)?esfinter(?:ec|o)tomia\b",
    "colocacion_stent": r"\b(?:se\s+(?:coloca|coloco|implanta|implanto)|colocamos|implantamos|procedemos\s+a\s+colocar)\s+(?:una?\s+)?(?:stent|protesis)\b",
    "retiro_stent": r"\b(?:se\s+(?:retira|retiro|extrae|extrajo)|retiramos|extraemos|procedemos\s+a\s+retirar)\s+(?:el\s+|la\s+|una?\s+)?(?:stent|protesis)\b",
    "extraccion_litos": r"\b(?:se\s+(?:retira|retiro|extrae|extrajo)|retiramos|extraemos|procedemos\s+a\s+retirar)\s+(?:(?:el|la|los|las|un|una|fragmentos|de)\s+)*(?:litos?|calculos?|barro|microlitos?)\b",
    "coledocoscopia": DO + r"(?:coledocoscopia|colangioscopia)\b",
    "litotripsia": r"\b(?:realizando|realizamos|realiza|realizo|efectua|efectuo)\s+(?:una\s+)?litotripsia\b",
    "dilatacion": r"\b(?:dilatamos|procedemos\s+a\s+dilatar)\s+(?:la\s+)?(?:papila|estenosis|esofago)\b|" + DO + r"(?:esfinteroplastia|dilatacion(?:es)?)\b",
    "hemostasia": DO + r"(?:escleroterapia|hemostasia)\b",
    "ligadura": DO + r"(?:endoligadura|ligadura)\b|\b(?:se\s+)?ligan\s+(?:las\s+)?varices\b|\b(?:se\s+procede\s+a\s+colocar|procedemos\s+a\s+colocar|se\s+colocan)\s+\d+\s+ligas\b",
    "puncion": r"\b(?:se\s+punciona|puncionamos|procedemos\s+a\s+puncionar)\b|" + DO + r"puncion\b",
    "drenaje": DO + r"drenaje\b|\b(?:se\s+)?drena(?:mos)?\b",
    "inyeccion": DO + r"(?:inyeccion|infiltracion)\b",
    "confocal": DO + r"(?:estudio|analisis)\b.{0,160}\bconfocal\b",
    "polipectomia_pinza": DO + r"polipectomia\s+con\s+(?:una\s+)?pinza(?:\s+de\s+biopsia)?\b|\b(?:se\s+(?:reseca|extrae)|retirado)\s+con\s+pinza\s+de\s+biopsia\b",
    "polipectomia_asa": DO + r"polipectomia\s+con\s+(?:una\s+)?(?:pinza\s+de\s+)?asa\b|\bse\s+reseca\s+con\s+asa\b",
}
ACT_PATTERNS = {k: re.compile(v) for k,v in ACT_PATTERNS.items()}
MENTIONS = re.compile(r"\b(?:stent|protesis|polipectomia|mucosectomia|biopsias?|esfinter(?:ec|o)tomia|"
    r"coledocoscopia|colangioscopia|litotripsia|dilatacion|endoligadura|ligadura|"
    r"puncion|escleroterapia|hemostasia|confocal|drenaje)\b")
MENTION_ACTS = {
    "stent": {"colocacion_stent","retiro_stent"}, "protesis": {"colocacion_stent","retiro_stent"},
    "polipectomia": {"polipectomia_pinza","polipectomia_asa"},
    "mucosectomia": set(), "esfinterectomia": {"esfinterotomia"},
    "esfinterotomia": {"esfinterotomia"}, "coledocoscopia": {"coledocoscopia"},
    "colangioscopia": {"coledocoscopia"}, "litotripsia": {"litotripsia"},
    "dilatacion": {"dilatacion"}, "endoligadura": {"ligadura"}, "ligadura": {"ligadura"},
    "puncion": {"puncion"}, "escleroterapia": {"hemostasia"}, "hemostasia": {"hemostasia"},
    "confocal": {"confocal"},
}


def active_sections(text):
    """Keep explicit history/plan scopes until a performed-section heading."""
    matches = list(SECTION.finditer(text))
    if not matches:
        return [text]
    parts = [text[:matches[0].start()]]
    for i, match in enumerate(matches):
        if match.group(1) in PERFORMED_SECTIONS:
            parts.append(text[match.end(): matches[i+1].start() if i+1<len(matches) else len(text)])
    return parts


def qualified(text, start, end):
    # Test the assertion locally, not every negative observation in a report.
    # Explicit periods delimit statements; newlines deliberately do not.
    prefix = re.split(r"[.!?;]", text[max(0,start-140):start])[-1]
    prefix = re.split(r"\b(?:finalmente|luego|concluimos)\b", prefix)[-1]
    tail = re.split(r"[.!?;]", text[end:end+120])[0]
    tail = re.split(r"\b(?:se\s+(?:introduce|observa|identifica)|concluimos|gastropatia|gastritis|hernia)\b",tail)[0]
    # Restrict tail qualifiers to temporal/negative assertions; ordinary 'sin
    # complicaciones' and 'sin lesiones' are findings, not denial of the act.
    return bool(CONTEXT_MODAL.search(prefix) or re.search(r"\b(?:no|nunca|tampoco|ni)\b",prefix)
        or DIRECT_DENIAL.search(prefix) or re.search(
        r"\b(?:si|cuando|manana|pendiente|previamente|anterior|programad[oa]|"
        r"cancelad[oa]|cancelo|suspendid[oa]|suspendio|ejemplo|intencion|dentro)\b|\b(?:no|sin)\s+(?:se\s+)?(?:realiz\w*|efectu\w*|tom\w*|exito)\b|"
        r"\b(?:en\s+(?:19|20)[0-9]{2}|hace\s+.{0,20}(?:anos?|meses?|dias?))\b",tail)
        or "¿" in prefix or "?" in text[start:end+len(tail)+1])


def biopsy_events(text):
    """Extract a local target after a performed sampling verb and instrument.

    The bounded target grammar handles unpunctuated source reports. It does not
    attach anatomy from another observation or a second intervention.
    """
    events=[]
    pattern=re.compile(r"\b(?:se\s+)?(?:toma(?:mos|n|ron)?|tomo|obtuvieron|obtuvo)\s+(?:una\s+)?biopsias?\b")
    for match in pattern.finditer(text):
        if qualified(text,match.start(),match.end()): continue
        following=text[match.end():match.end()+400]
        # Instrument vocabulary cannot be an arbitrary clinical assertion.
        target=re.search(r"\b(?:de|del|en)\s+(?:(?:la|el|mucosa)\s+)*(?:antro|cuerpo|duodeno|bulbo|esofago|colon|recto|ileon|sigmoides?|coledoco|pancreas|papila|higado|hepatico|lesion|adenopatia)\b",following)
        if not target: continue
        prefix=set(re.findall(r"[a-z]+",following[:target.start()]))
        if prefix - set("con pinza de biopsia endoscopica endoscopico descartable desechable estandar esteril escalonadas una un el la pediatrica para toma muestras s sin aguja canal trabajo mm cm long longitud".split()): continue
        tail=re.split(r"[.!?;]",following[target.start():],maxsplit=1)[0]
        # Source narratives often omit punctuation before the conclusion. Only
        # these explicit clinical boundaries can end an act; arbitrary words
        # ('como ejemplo', a future qualifier, etc.) cannot be truncated away.
        tail=re.split(r"\b(?:gastritis|gastropatia|hernia|pandiverticulosis|diverticulosis|endoscopicamente|"
                      r"ileocolonoscopia\s+sin\s+alteraciones|"
                      r"concluimos|se\s+introduce|se\s+observa|estudio\s+suspendido)\b",tail,maxsplit=1)[0]
        allowed=set("de del la el las los y e en antro antral cuerpo corporal duodeno duodenal bulbo esofago colon colonica colonico recto rectal ileon terminal sigmoide sigmoides ascendente descendente transverso mucosa gastrica oxintica polipo polipos gastricos coledoco pancreas pancreatica papila vater hepatico comun extrahepatico wirsung cabeza higado derecho izquierdo lesion tumoral subepitelial adenopatia perigastrica para estudio histopatologico histopatologia histologico histologia anatomia patologica patologia analisis descartar procesos inflamatorios microscopicos investigacion uno dos tres cuatro muestras fragmentos sin complicaciones inmediatas".split())
        if set(re.findall(r"[a-z]+",tail))-allowed: continue
        sites=site_labels(tail)
        if sites: events.append(sites)
    return events


def clinical_features(payload):
    acts, families, uncertain = set(), set(), set()
    biopsy_sites = set()
    for field in ("hallazgos_conclusion", "descripcion_estudio_013"):
        raw = _normalize_findings(payload.get(field))
        for text in active_sections(raw):
            # Context is accepted only from factual segments, never the system name.
            context = text
            if re.search(r"\bno\s+(?:se\s+)?(?:realizo|realiza|efectuo|practico)\s+(?:la\s+)?(?:cpre|ecoendoscopia)\b|\b(?:cpre|ecoendoscopia)\s+(?:cancelada|suspendida)\b",text):
                uncertain.add("procedimiento_contradictorio")
            if re.search(r"\b(?:cpre|duodenoscopio)\b", context): families.add("cpre")
            if re.search(r"\b(?:ecoendoscopio|ecoendoscopia|ultrasonido\s+endoscopico)\b", context): families.add("eco")
            if re.search(r"\b(?:colonoscopi[ao]|videocolonoscopi[ao]|ileocolonoscopia|via\s+(?:anal|rectal))\b", context): families.add("colon")
            if re.search(r"\b(?:via\s+oral|gastroscopia|endoscopia(?:\s+digestiva)?\s+alta)\b", context): families.add("alta")
            support = _biopsy_support(text)
            for code,family in (("70200003","alta"),("70200004","colon")):
                if support[code]:
                    if _biopsy_denied(text, code) or _procedure_denied(text, code):
                        uncertain.add("biopsia_contradictoria")
                    else:
                        acts.add("biopsia_"+family)
                        families.add(family)
                        for event in support[code]:
                            biopsy_sites.update(site_labels(event["texto_normalizado"]))
            for key,pattern in ACT_PATTERNS.items():
                for match in pattern.finditer(text):
                    if qualified(text,match.start(),match.end()): uncertain.add("acto_calificado")
                    else: acts.add(key)
            for match in re.finditer(r"\b(?:bajo\s+)?(?:guia|vision|control)\s+fluoroscopic[oa]\b",text):
                if not qualified(text,match.start(),match.end()): acts.add("fluoroscopia")
            for sites in biopsy_events(text):
                biopsy_sites.update(sites)
                if sites & {"lesion_tumoral","lesion_subepitelial","lesion_papila_vater","adenopatia_perigastrica"}: acts.add("biopsia_lesion")
                elif sites & {"coledoco","pancreas","papila","hepatico_comun","extrahepatico","wirsung"}: acts.add("biopsia_biliar")
                elif sites & {"higado","hepatico_derecho","hepatico_izquierdo"}: acts.add("biopsia_hepatica")
                elif sites & {"colon","recto","ileon","sigmoide"} and "colon" in families: acts.add("biopsia_colon")
                elif sites & {"antro","cuerpo","duodeno","bulbo","esofago","mucosa_gastrica","polipos_gastricos"} and "alta" in families: acts.add("biopsia_alta")
            for match in re.finditer(r"\b(?:se\s+)?(?:toma(?:mos|n|ron)?|tomo|obtuvieron|obtuvo)\s+(?:una\s+)?biopsias?\b",text):
                if qualified(text,match.start(),match.end()): uncertain.add("biopsia_calificada")
            if "eco" in families and re.search(r"\bse\s+introduce\b.{0,40}\becoendoscopio\b",text): acts.add("exploracion_eco")
            for family,code in (("alta","70200003"),("colon","70200004")):
                if "biopsia_"+family in acts and (_biopsy_denied(text,code) or _procedure_denied(text,code)):
                    uncertain.add("biopsia_contradictoria")
            # A negative-only biopsy cannot support an alternative; contradictory
            # or unresolved intervention language must not yield a simpler combo.
            if re.search(r"\bbiopsias?\b", text) and not any(a.startswith("biopsia_") for a in acts):
                uncertain.add("biopsia_no_resuelta")
            for match in MENTIONS.finditer(text):
                mention=match.group()
                if mention in MENTION_ACTS:
                    if mention=="dilatacion" and re.search(r"(?:gran|una)\s*$",text[max(0,match.start()-20):match.start()]) and re.match(r"\s+del\s+coledoco\b",text[match.end():]):
                        continue  # observed duct caliber, not an intervention
                    if qualified(text,match.start(),match.end()):
                        uncertain.add("intervencion_calificada:"+mention)
                    elif not acts & MENTION_ACTS[mention]:
                        uncertain.add("intervencion_no_resuelta:"+mention)
    if "cpre" in families: families.discard("alta")
    if "eco" in families: families.discard("alta")
    return {"familias": sorted(families), "actos": sorted(acts),
            "sitios_biopsia": sorted(biopsy_sites), "incertidumbres": sorted(uncertain)}


SITE_PATTERNS = {
    "antro": r"\b(?:antro|antral)\b", "cuerpo": r"\b(?:cuerpo|corporal)\b",
    "duodeno": r"\bduodeno\b", "bulbo": r"\bbulbo\b", "esofago": r"\besofago\b",
    "polipos_gastricos": r"\bpolipos?\s+gastricos?\b",
    "mucosa_gastrica": r"\bmucosa\s+(?:gastrica|oxintica)\b",
    "colon": r"\bcolon(?:ica|ico)?\b", "recto": r"\brecto\b", "ileon": r"\bileon\b",
    "sigmoide": r"\bsigmoid(?:e|es)\b", "coledoco": r"\bcoledoco\b",
    "pancreas": r"\bpancrea(?:s|tica|tico)\b", "papila": r"\bpapila\b",
    "hepatico_comun": r"\bhepatico\s+comun\b", "extrahepatico": r"\bextrahepatico\b",
    "wirsung": r"\bwirsung\b", "higado": r"\bhigado\b",
    "hepatico_derecho": r"\bhepatico\s+derecho\b", "hepatico_izquierdo": r"\bhepatico\s+izquierdo\b",
    "lesion_tumoral": r"\blesion\s+tumoral\b", "lesion_subepitelial": r"\blesion\s+subepitelial\b",
    "lesion_papila_vater": r"\blesion\s+de\s+papila\s+de\s+vater\b",
    "adenopatia_perigastrica": r"\badenopatia\s+perigastrica\b",
}


def site_labels(text):
    normalized = _normalize_findings(text)
    sites={key for key,pattern in SITE_PATTERNS.items() if re.search(pattern, normalized)}
    if "lesion_papila_vater" in sites:
        sites.discard("papila")
    return sites
