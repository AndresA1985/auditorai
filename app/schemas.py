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


class CodigoScore(BaseModel):
    codigo: str
    score: float
    selected: bool = False


class PlantillaScore(BaseModel):
    cod_plantilla: str
    descripcion: str = Field(default='')
    desc_comp: str = Field(default='')
    codigos: List[str] = Field(default_factory=list)
    score: float


class PrediccionPayload(BaseModel):
    codigos: List[str]
    codigo_scores: Dict[str, float] = Field(default_factory=dict)
    codigo_ranking: List[CodigoScore] = Field(default_factory=list)
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