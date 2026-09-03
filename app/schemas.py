from datetime import date
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
