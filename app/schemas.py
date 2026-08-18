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


class PrediccionPayload(BaseModel):
    codigos: List[str]
    codigo_scores: Dict[str, float] = Field(default_factory=dict)
    honorarios_codigo: Dict[str, str]
    honorario: str
    tiempo_anestesia: Union[int, str, None] = None
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