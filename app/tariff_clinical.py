"""Conservative performed-biopsy evidence, shared by Excel and PDF proposals."""
import html
import re
import unicodedata

PAIR = frozenset({"70200003", "70200004"})

BIOPSY_ACTION = re.compile(
    r"\b(?:se\s+)?(?:toma(?:n|ron|mos)?|tomo|realiza(?:n|ron|mos)?|realizo|"
    r"obtiene(?:n)?|obtuvo|obtuvieron|extrae(?:n)?|extrajo|extrajeron|"
    r"practica(?:n)?|practico)\s+(?:una\s+|varias\s+|[0-9]+\s+)?biopsias?\b"
)
NON_PERFORMED = re.compile(
    r"\b(?:no|nunca|tampoco|sin|ni)\b|"
    r"\b(?:plan(?:ea|ifica|ificada|ear|ificar)?|programa(?:da|r)?|pendiente|"
    r"recomienda|sugiere|solicita|indica|propuesta|antecedente|antecedentes|"
    r"previo|previa|anterior|historico|hipotetico|si)\b|"
    r"\bhace\b.{0,60}\b(?:anos?|meses?|dias?|semanas?)\b|\ben\s+(?:19|20)[0-9]{2}\b"
)
NON_FACTUAL_TAIL = re.compile(
    r"\b(?:no|nunca|tampoco|si|cuando|manana|proxim[oa]s?|previst[oa]s?|"
    r"programad[oa]s?|pendiente|eventual|posible|necesari[oa]|planificad[oa]s?|"
    r"podria|podra|debe|debera|recomendada|solicitada)\b|"
    r"\ben\s+caso\b|\bpor\s+realizar\b|\bhace\b.{0,60}\b(?:anos?|meses?|dias?|semanas?)\b"
)
HISTORICAL_OR_TEMPORAL = re.compile(
    r"\b(?:previ[oa]s?|previamente|anterior(?:es)?|antecedentes?|historic[oa])\b|"
    r"\b(?:hace|en|dentro\s+de)\s+(?:\w+\s+){0,4}(?:anos?|meses?|dias?|semanas?)\b|"
    r"\ben\s+(?:19|20)[0-9]{2}\b"
)
SECTION = re.compile(
    r"(?:^|[.\n;])\s*(plan(?:\s+de\s+tratamiento)?|planificacion|recomendacion(?:es)?|indicaciones|"
    r"solicitud|antecedentes(?:\s+\w+){0,3}|historia(?:\s+\w+){0,3}|insumos|"
    r"hallazgos|tecnica|procedimiento\s+realizado|descripcion\s+del\s+procedimiento)\s*:",
)
PERFORMED_SECTIONS = {"hallazgos", "tecnica", "procedimiento realizado", "descripcion del procedimiento"}
BIOPSY_BOUNDARY = re.compile(
    r"\b(?:estudio\s+suspendido|se\s+introduce|"
    r"(?:colonoscopia|endoscopia(?:\s+digestiva)?\s+alta)\s*:)"
)
UPPER_SITE = re.compile(r"\b(?:antro|fundus|estomago|esofago|duodeno|bulbo|gastric[oa]s?)\b")
LOWER_SITE = re.compile(r"\b(?:colon|colonica|colonico|ciego|recto|sigmoides|sigmoide|ileon)\b")
SITE_LINK = re.compile(
    r"\b(?:de|del|en)\s+(?:(?:la|el|mucosa|segmento)\s+)*"
    r"(?:antro|fundus|estomago|esofago|duodeno|bulbo|gastric[oa]s?|"
    r"colon|ciego|recto|sigmoides|sigmoide|ileon)\b"
)
# Only this bounded instrument/target grammar may intervene before the site.
# Arbitrary subsequent anatomy, clinical sentences, plans, etc. cannot attach.
BIOPSY_PREFIX_WORDS = frozenset((
    "con una un pinza de del la el biopsia estandar long longitud s c sin aguja "
    "canal trabajo mm cm esteril descartable desechable mucosa tejido lesion "
    "polipo fragmento fragmentos"
).split())
# After the site, accept anatomy, sampling counts and specimen destination.
# Any other free clinical assertion makes this event unresolved. This keeps
# unrecognised modality/denial from becoming positive evidence by omission.
BIOPSY_TARGET_WORDS = frozenset((
    "de del la el las los y e en un una uno dos tres cuatro cinco seis "
    "antro antral cuerpo corporal fundus fondo gastrico gastrica gastricos gastricas "
    "estomago esofago esofagico esofagica duodeno duodenal bulbo mucosa segmento "
    "colon colonica colonico ciego recto sigmoides sigmoide ileon ascendente "
    "descendente transverso proximal distal derecho izquierda izquierdo derecha "
    "terminal rectal rectosigmoide muestras muestra fragmento fragmentos "
    "para estudio histopatologico histopatologia histologico histologia "
    "anatomia patologica patologia analisis mm cm"
).split())
BIOPSY_DENIAL = re.compile(
    r"\bbiopsias?\s+no\s+(?:(?:fue|fueron)\s+)?(?:realizad[ao]s?|tomad[ao]s?|obtenid[ao]s?|efectuad[ao]s?)\b|"
    r"\bno\s+(?:se\s+)?(?:realiza(?:n|ron)?|realizo|toma(?:n|ron)?|tomo|obtiene(?:n)?|obtuvo|obtuvieron)\s+biopsias?\b"
)
UPPER_NAME = r"(?:endoscopia(?:\s+digestiva)?\s+alta|gastroscopia|esofagogastroduodenoscopia)"
LOWER_NAME = r"(?:colonoscopia|videocolonoscopia)"


def _normalize_findings(value):
    if not isinstance(value, str):
        return ""
    # Decode encoded whitespace before stripping markup; keep section boundaries.
    text = html.unescape(html.unescape(value))
    text = re.sub(r"<\s*(?:br\b[^>]*|/p\s*|/div\s*)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]*>", " ", text)
    text = "".join(char for char in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(char))
    return re.sub(r"[^\S\n]+", " ", text).strip()


def _procedure_context(text, code):
    # A planned/negated heading elsewhere is not evidence of the performed
    # procedure. Only affirmative context inside this active section is used.
    statements = re.split(r"[.!?;\n]", text)
    text = " ".join(statement for statement in statements
                    if not NON_PERFORMED.search(statement) and not NON_FACTUAL_TAIL.search(statement))
    if code == "70200003":
        explicit = re.search(r"\b(?:" + UPPER_NAME + r"|via\s+oral)\b", text)
        landmarks = set(re.findall(r"\b(?:esofago|estomago|duodeno|bulbo|antro)\b", text))
        return bool(explicit or len(landmarks) >= 2)
    return bool(re.search(r"\b(?:" + LOWER_NAME + r"|(?:video)?colonoscopio|via\s+anal)\b", text))


def _procedure_denied(text, code):
    name = UPPER_NAME if code == "70200003" else LOWER_NAME
    return bool(re.search(
        r"\bno\s+(?:se\s+)?(?:realizo|realiza|practico|efectuo)\s+(?:la\s+)?" + name + r"\b|"
        r"\b" + name + r"\s+(?:(?:fue|se\s+encuentra|se)\s+)?(?:suspendida|suspende|cancelada)\b",
        text,
    ))


def _biopsy_denied(text, code):
    for clause in re.split(r"[.!?;\n]", text):
        if not BIOPSY_DENIAL.search(clause):
            continue
        sites = {candidate for candidate, pattern in (("70200003", UPPER_SITE), ("70200004", LOWER_SITE))
                 if pattern.search(clause)}
        if not sites or code in sites:
            return True
    return False


def _biopsy_support(text):
    """Use affirmative act + local biopsy site, bounded before the next event.

    Punctuation inside an instrument specification is not a clinical boundary:
    the user's note has `2.8 . Esteril. DescartableDE ANTRO` after the act.
    A generic suspension after that completed biopsy does not erase the act.
    """
    support = {code: [] for code in PAIR}
    actions = list(BIOPSY_ACTION.finditer(text))
    sections = list(SECTION.finditer(text))
    for index, action in enumerate(actions):
        previous_sections = [section for section in sections if section.end() <= action.start()]
        if previous_sections and previous_sections[-1].group(1) not in PERFORMED_SECTIONS:
            continue
        section_start = previous_sections[-1].end() if previous_sections else 0
        section_end = next((section.start() for section in sections if section.start() > action.start()), len(text))
        prefix = text[max(section_start, action.start() - 180):action.start()]
        prefix = re.split(r"[.!?;]", prefix)[-1]
        if NON_PERFORMED.search(prefix) or HISTORICAL_OR_TEMPORAL.search(prefix) or "¿" in prefix:
            continue
        end = min(section_end, action.end() + 600)
        if index + 1 < len(actions):
            end = min(end, actions[index + 1].start())
        boundary = BIOPSY_BOUNDARY.search(text, action.end(), end)
        if boundary:
            end = boundary.start()
        following = text[action.end():end].replace("descartablede ", "descartable de ")
        # Preserve modality/negation in the event; truncating at those words
        # would hide contradictions such as `de antro pero no se realizo`.
        linked = SITE_LINK.search(following)
        if linked is None:
            continue
        prefix_words = set(re.findall(r"[a-z]+", following[:linked.start()]))
        if prefix_words - BIOPSY_PREFIX_WORDS:
            continue
        # Decimal points/periods are tolerated only in the approved prefix.
        # Once the site begins, the next statement is a hard event boundary.
        event_parts = re.split(r"([.!?;])", following[linked.start():], maxsplit=1)
        target = event_parts[0]
        if (NON_FACTUAL_TAIL.search(target) or HISTORICAL_OR_TEMPORAL.search(target)
                or (len(event_parts) > 1 and event_parts[1] == "?")):
            continue
        if set(re.findall(r"[a-z]+", target)) - BIOPSY_TARGET_WORDS:
            continue
        site_codes = [code for code, site in (("70200003", UPPER_SITE), ("70200004", LOWER_SITE))
                      if site.search(target)]
        quote = text[action.start():action.end()] + following[:linked.start()] + target
        for code in site_codes:
            if _procedure_context(text[section_start:section_end], code):
                support[code].append({"campo": "hallazgos_conclusion",
                    "texto_normalizado": quote.strip()[:400]})
    return support
