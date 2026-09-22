from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


# INICIO CAMBIO AUDITORIA TARIFARIO: contrato documental aditivo y separado del ranking clínico.
class CodigoTarifarioDocumental(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codigo: str = Field(pattern=r"^[0-9]{5,8}$")
    descripcion: str = Field(default="", max_length=180)
    origen: Literal["texto", "ocr"]
    pagina: int = Field(ge=1, le=10, strict=True)
    confianza_ocr: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Confianza de reconocimiento de caracteres OCR; no es ranking "
            "ni probabilidad clínica."
        ),
    )

    @model_validator(mode="after")
    def ocr_score_is_not_a_model_score(self):
        if self.confianza_ocr is not None and self.origen != "ocr":
            raise ValueError("La confianza OCR corresponde solo a texto OCR")
        return self


AdvertenciaTarifario = Literal[
    "sin_documento",
    "sin_archivo",
    "pdf_no_adjuntado",
    "pdf_encriptado",
    "pdf_sin_paginas",
    "tiempo_extraccion_agotado",
    "ocr_no_disponible",
    "ocr_sin_codigos_verificables",
    "requiere_verificacion_ocr",
    "tiempo_ocr_agotado",
    "ocr_fallido",
    "limite_codigos",
    "sin_codigos_tarifarios",
    "pdf_invalido",
    "extraccion_fallida",
    "formato_no_identificado",
    "tabla_ambigua",
    "documento_estructura_ambigua",
    "codigo_validacion_no_verificable",
    "codigo_validacion_requiere_revision",
    "celda_tarifario_no_verificable",
]
MotivoAbstencionTarifario = Literal[
    "sin_features_utilizables",
    "sin_codigos_sobre_umbral",
]


class EvidenciaTarifario(BaseModel):
    # Only bounded structured references cross this boundary, never PDF/OCR bodies.
    model_config = ConfigDict(extra="forbid")

    estado: Literal[
        "extraido", "sin_codigos", "ilegible", "sin_documento", "sin_archivo", "sin_pdf"
    ]
    fuente: Literal["texto", "ocr", "mixta", "ninguna"]
    documento_sha256: Optional[str] = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    codigos: List[CodigoTarifarioDocumental] = Field(default_factory=list, max_length=25)
    advertencias: List[AdvertenciaTarifario] = Field(default_factory=list, max_length=25)

    @model_validator(mode="after")
    def source_matches_references(self):
        if (self.estado == "extraido") != bool(self.codigos):
            raise ValueError("El estado documental debe corresponder a los códigos extraídos")
        origins = {item.origen for item in self.codigos}
        expected = "mixta" if len(origins) > 1 else next(iter(origins), "ninguna")
        if self.fuente != expected:
            raise ValueError("La fuente documental debe corresponder al origen de las referencias")
        document_present = self.estado in {"extraido", "sin_codigos", "ilegible"}
        if document_present and self.documento_sha256 is None:
            raise ValueError("La evidencia de un PDF presente requiere su hash SHA-256")
        if not document_present and self.documento_sha256 is not None:
            raise ValueError("Un estado sin PDF no puede declarar hash documental")
        return self


class CitaTarifario(BaseModel):
    codigo: str = Field(pattern=r"^[0-9]{5,8}$")
    fuente: Literal[
        "documento_texto",
        "documento_ocr",
        "hallazgos_conclusion",
        "descripcion_estudio_013",
        "informe_tecnico_justificacion",
    ]
    campo: str
    texto: str = Field(min_length=1, max_length=500)
    pagina: Optional[int] = Field(default=None, ge=1, le=10)
    documento_sha256: Optional[str] = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    verificada: bool
    alcance: Literal["contexto_extraido_gen", "clinica_literal"]


class DiscrepanciaTarifario(BaseModel):
    codigo: str = Field(pattern=r"^[0-9]{5,8}$")
    tipo: Literal[
        "adicional_modelo", "ausente_modelo", "sin_evidencia_documental", "contradiccion_clinica"
    ]
    motivo: str


class JustificacionTarifario(BaseModel):
    codigo: str = Field(pattern=r"^[0-9]{5,8}$")
    en_documento: bool
    seleccionado: bool
    citas: List[CitaTarifario] = Field(default_factory=list)
    requiere_revision: bool
    justificacion: str


class ComparacionRankingTarifario(BaseModel):
    codigo: str = Field(pattern=r"^[0-9]{5,8}$")
    score_ranking: float = Field(
        description="Score original del modelo, no calibrado; no se repondera por el PDF."
    )
    selected: bool
    en_documento: bool
    clasificacion: Literal[
        "coincidencia", "documental_no_seleccionado", "adicional", "sin_referencia_documental"
    ]
    citas: List[CitaTarifario] = Field(default_factory=list)
    semantica_score: str


class OpcionTarifarioDocumental(BaseModel):
    codigo: str = Field(pattern=r"^[0-9]{5,8}$")
    descripcion: str = Field(default="", max_length=180)
    prioridad: Literal["alta"]
    requiere_revision: bool
    etiqueta: str
    fuente: Literal["documento_codigo_validacion"]
    origen: Literal["texto", "ocr"]
    pagina: int = Field(ge=1, le=10)
    documento_sha256: Optional[str] = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    confianza_ocr: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Reconocimiento de caracteres OCR, independiente del modelo.",
    )
    modelo_score: Optional[float] = Field(
        default=None,
        description="Score original si existe en ranking; None si no existe.",
    )
    modelo_selected: bool
    coincidencia_modelo: bool
    citas: List[CitaTarifario] = Field(default_factory=list)
    motivo: str

    @model_serializer(mode="wrap")
    def preserve_absent_model_score(self, handler):
        serialized = handler(self)
        serialized.setdefault("modelo_score", None)
        return serialized
# FIN CAMBIO AUDITORIA TARIFARIO: contrato documental aditivo y separado del ranking clínico.


class PrediccionRequest(BaseModel):
    id_agenda: int
    fecha_agenda: Optional[str] = None
    cedula: Optional[str] = None
    paciente: Optional[str] = None
    sexo: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    edad: Optional[Any] = None
    id_hc: Optional[Any] = None
    id_empresa: Optional[str] = None
    id_seguro: Optional[Any] = None
    mes_plano: Optional[Any] = None
    procedimiento_sistema: Optional[str] = None
    hallazgos_conclusion: Optional[str] = None
    descripcion_estudio_013: Optional[str] = None
    # INICIO CAMBIO AUDITORIA TARIFARIO: evidencia estructurada, sin texto PDF crudo ni identidad.
    evidencia_tarifario: Optional[EvidenciaTarifario] = None
    informe_tecnico_justificacion: Optional[str] = Field(default=None, max_length=20000)
    # FIN CAMBIO AUDITORIA TARIFARIO: evidencia estructurada, sin texto PDF crudo ni identidad.


class EvidenciaRanking(BaseModel):
    disponible: bool = False
    texto_soporte: Optional[str] = None
    justificacion: str
    score_ranking: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    porcentaje_ranking: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    semantica_score: str = (
        "Puntaje relativo de ranking; no es F1 ni probabilidad calibrada."
    )
    fuente: str
    valor_sugerido: Optional[str] = None
    unidad: Optional[str] = None


class EvidenciasCodigo(BaseModel):
    codigo: EvidenciaRanking
    honorario: EvidenciaRanking
    tiempo_anestesia: EvidenciaRanking


class CodigoScore(BaseModel):
    codigo: str
    score: float = Field(description=(
        "Puntaje usado exclusivamente para ordenar codigos sugeridos. "
        "No es F1, confianza ni una probabilidad calibrada."
    ))
    selected: bool = False
    descripcion_codigo: str = ""
    texto_soporte: Optional[str] = None
    justificacion: str = Field(
        default="No se encontro evidencia textual suficiente para justificar este codigo."
    )
    evidencias: Optional[EvidenciasCodigo] = None


class MetricasModelo(BaseModel):
    f1_macro: float = Field(ge=0.0, le=1.0)
    f1_weighted: float = Field(ge=0.0, le=1.0)
    version_modelo: str
    conjunto_evaluacion: str
    fecha_evaluacion: date
    cantidad_muestras: int = Field(ge=1)


class PlantillaScore(BaseModel):
    cod_plantilla: str
    descripcion: str = Field(default="")
    desc_comp: str = Field(default="")
    codigos: List[str] = Field(default_factory=list)
    score: float


class PrediccionPayload(BaseModel):
    codigos: List[str]
    codigo_scores: Dict[str, float] = Field(default_factory=dict)
    codigo_ranking: List[CodigoScore] = Field(default_factory=list)
    metricas_modelo: Optional[MetricasModelo] = None
    plantilla_ranking: List[PlantillaScore] = Field(default_factory=list)
    honorarios_codigo: Dict[str, str]
    honorario: str
    tiempo_anestesia: Union[int, str, None] = None
    tiempos_anestesia_codigo: Dict[str, Union[int, str]] = Field(default_factory=dict)
    nombre_procedimiento: str = ""
    observacion_auditor: str
    # INICIO CAMBIO AUDITORIA TARIFARIO: comparación posterior y prioridad documental separada.
    evidencia_tarifario: Optional[EvidenciaTarifario] = None
    requiere_revision: Optional[bool] = None
    discrepancias: Optional[List[DiscrepanciaTarifario]] = None
    citas_documentales: Optional[List[CitaTarifario]] = None
    justificaciones_tarifario: Optional[List[JustificacionTarifario]] = None
    comparacion_ranking_tarifario: Optional[List[ComparacionRankingTarifario]] = None
    opciones_tarifario_documental: Optional[List[OpcionTarifarioDocumental]] = None
    abstencion: Optional[bool] = None
    motivo_abstencion: Optional[MotivoAbstencionTarifario] = None
    feature_schema_version: Optional[str] = None
    feature_mode: Optional[Literal["clinical", "document_only", "clinical_document"]] = None
    # FIN CAMBIO AUDITORIA TARIFARIO: comparación posterior y prioridad documental separada.

    # INICIO CAMBIO AUDITORIA TARIFARIO: invariantes serializables de abstención y ranking intacto.
    @model_validator(mode="after")
    def validate_tariff_invariants(self):
        if self.abstencion is True:
            if self.codigos or self.requiere_revision is not True or self.motivo_abstencion is None:
                raise ValueError("La abstención exige códigos vacíos, revisión y motivo tipado")
        elif self.motivo_abstencion is not None:
            raise ValueError("El motivo de abstención sólo aplica cuando abstencion=true")

        if self.comparacion_ranking_tarifario is not None:
            ranking = {item.codigo: item for item in self.codigo_ranking}
            if {item.codigo for item in self.comparacion_ranking_tarifario} != set(ranking):
                raise ValueError("La comparación documental debe cubrir el ranking original")
            for item in self.comparacion_ranking_tarifario:
                original = ranking[item.codigo]
                if item.score_ranking != original.score or item.selected != original.selected:
                    raise ValueError("La comparación documental no puede alterar score ni selected")
        return self
    # FIN CAMBIO AUDITORIA TARIFARIO: invariantes serializables de abstención y ranking intacto.


class PrediccionResponse(BaseModel):
    ok: bool = True
    mensaje: str
    prediccion: PrediccionPayload


class CodigoClase(BaseModel):
    codigo: str
    frecuencia: int
    agendas: int
    porcentajes_usados: str = ""
    tiempos_anestesia_usados: str = ""
    clase_frecuencia: str
