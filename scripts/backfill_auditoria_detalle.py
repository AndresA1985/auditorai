import argparse
import os
import json
import re
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db import db_connection, fetch_all


def required_source_env(name: str, *, allow_empty: bool = False) -> str:
    value = os.getenv(name)
    if value is None or (value == "" and not allow_empty):
        raise RuntimeError("Falta configurar {0} en el archivo .env".format(name))
    return value


def source_db_config() -> dict:
    return {
        "host": required_source_env("SOURCE_DB_HOST"),
        "port": int(required_source_env("SOURCE_DB_PORT")),
        "user": required_source_env("SOURCE_DB_USERNAME"),
        "password": required_source_env("SOURCE_DB_PASSWORD", allow_empty=True),
        "database": required_source_env("SOURCE_DB_DATABASE"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": True,
    }


def fetch_all_source(sql: str, params=None):
    conn = pymysql.connect(**source_db_config())
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or {})
            return cursor.fetchall()
    finally:
        conn.close()


def normalizar_codigo(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\D+", "", str(value))


def normalizar_numero(value) -> str:
    if value is None:
        return ""
    raw = str(value).strip().replace(",", ".")
    if not raw:
        return ""
    try:
        number = Decimal(raw)
    except InvalidOperation:
        digits = re.sub(r"\D+", "", raw)
        return digits
    if number == number.to_integral_value():
        return str(int(number))
    return str(number.normalize())


def parse_honorarios(value) -> dict[str, str]:
    if value is None:
        return {}
    pairs = {}
    for code, percentage in re.findall(r"(\d+)\s*\(\s*([\d,.]+)\s*\)", str(value)):
        normalized_code = normalizar_codigo(code)
        normalized_percentage = normalizar_numero(percentage)
        if normalized_code and normalized_percentage:
            pairs[normalized_code] = normalized_percentage
    return pairs


def normalizar_scalar(value) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not re.fullmatch(r"[0-9,.]+", raw):
        return ""
    return normalizar_numero(raw)


def placeholders(values) -> str:
    return ",".join(["%s"] * len(values))


def build_filters(args):
    filters = ["aadd.estado = 1"]
    params = []
    if args.mes_plano:
        filters.append("aadd.mes_plano = %s")
        params.append(args.mes_plano)
    if args.id_agenda:
        filters.append("aadd.id_agenda = %s")
        params.append(args.id_agenda)
    if args.id_empresa:
        filters.append("aadd.id_empresa = %s")
        params.append(args.id_empresa)
    return " AND ".join(filters), params


def cargar_destino(args) -> list[dict]:
    where_sql, params = build_filters(args)
    sql = f"""
        SELECT
            aadd.id,
            aadd.id_agenda,
            aadd.mes_plano,
            aadd.id_empresa,
            aadd.codigo,
            aadd.codigo_grupo_auditor,
            aadd.tiempo_anestesia,
            aadd.porcentaje,
            aadd.honorario_auditor
        FROM ap_auditoria_doctor_detalle aadd
        WHERE {where_sql}
          AND aadd.codigo IS NOT NULL
          AND aadd.codigo <> ''
        ORDER BY aadd.id_agenda ASC, aadd.id ASC
    """
    if args.limit:
        sql += "\nLIMIT {0}".format(int(args.limit))
    rows = fetch_all(sql, params)
    for row in rows:
        row["codigo_norm"] = normalizar_codigo(row.get("codigo"))
        row["honorarios_parseados"] = parse_honorarios(row.get("honorario_auditor"))
    return [row for row in rows if row["codigo_norm"]]


def cargar_tiempos_archivo_plano(rows: list[dict]) -> dict[tuple[str, str, str, str], str]:
    agenda_values = sorted({str(row["id_agenda"]) for row in rows if row.get("id_agenda") is not None})
    if not agenda_values:
        return {}

    result = {}
    chunk_size = 500
    for start in range(0, len(agenda_values), chunk_size):
        chunk = agenda_values[start:start + chunk_size]
        sql = f"""
            SELECT
                h.id_agenda,
                apc.mes_plano,
                apc.id_empresa,
                apd.codigo,
                apd.cantidad
            FROM archivo_plano_cabecera apc
            LEFT JOIN historiaclinica h ON h.hcid = apc.id_hc
            JOIN archivo_plano_detalle apd
              ON apd.id_ap_cabecera = apc.id
             AND apd.estado = 1
            WHERE apc.estado = 1
              AND h.id_agenda IN ({placeholders(chunk)})
              AND (
                    apd.tipo = 'TA'
                    OR apd.descripcion LIKE '%%TIEMPO DE ANESTESIA%%'
              )
              AND apd.codigo IS NOT NULL
              AND apd.codigo <> ''
              AND apd.codigo <> 0
        """
        for row in fetch_all_source(sql, chunk):
            code = normalizar_codigo(row.get("codigo"))
            time_value = normalizar_numero(row.get("cantidad"))
            if not code or not time_value:
                continue
            key = (str(row.get("id_agenda")), str(row.get("mes_plano")), str(row.get("id_empresa")), code)
            result[key] = time_value
    return result


def armar_actualizaciones(rows: list[dict], tiempos: dict[tuple[str, str, str, str], str], args) -> list[dict]:
    updates = []
    stats = defaultdict(int)
    for row in rows:
        code = row["codigo_norm"]
        key = (str(row.get("id_agenda")), str(row.get("mes_plano")), str(row.get("id_empresa")), code)
        nuevo_tiempo = tiempos.get(key)
        nuevo_honorario = row["honorarios_parseados"].get(code)

        if args.require_source_time and not nuevo_tiempo:
            stats["sin_ta_fuente"] += 1
            continue

        if not nuevo_tiempo:
            nuevo_tiempo = normalizar_numero(row.get("tiempo_anestesia"))
        if not nuevo_honorario:
            nuevo_honorario = normalizar_scalar(row.get("porcentaje"))
        if not nuevo_honorario:
            nuevo_honorario = normalizar_scalar(row.get("honorario_auditor"))

        changes = {}
        if nuevo_tiempo and nuevo_tiempo != normalizar_numero(row.get("tiempo_anestesia")):
            changes["tiempo_anestesia"] = nuevo_tiempo
        if nuevo_honorario and nuevo_honorario != normalizar_scalar(row.get("honorario_auditor")):
            changes["honorario_auditor"] = nuevo_honorario

        if changes:
            updates.append({
                "id": row["id"],
                "id_agenda": row.get("id_agenda"),
                "mes_plano": row.get("mes_plano"),
                "codigo": code,
                "antes": {
                    "tiempo_anestesia": row.get("tiempo_anestesia"),
                    "porcentaje": row.get("porcentaje"),
                    "honorario_auditor": row.get("honorario_auditor"),
                },
                "despues": changes,
            })
            for field in changes:
                stats["cambia_" + field] += 1
        else:
            stats["sin_cambios"] += 1
    return updates, dict(stats)


def aplicar_actualizaciones(updates: list[dict]) -> None:
    with db_connection() as conn:
        with conn.cursor() as cursor:
            for item in updates:
                fields = []
                params = []
                for field in ("tiempo_anestesia", "honorario_auditor"):
                    if field in item["despues"]:
                        fields.append(f"{field} = %s")
                        params.append(item["despues"][field])
                if not fields:
                    continue
                params.append(item["id"])
                cursor.execute(
                    "UPDATE ap_auditoria_doctor_detalle SET {0} WHERE id = %s".format(", ".join(fields)),
                    params,
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normaliza tiempo_anestesia y honorario_auditor por codigo en ap_auditoria_doctor_detalle."
    )
    parser.add_argument("--mes-plano", default=None)
    parser.add_argument("--id-agenda", default=None)
    parser.add_argument("--id-empresa", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Ejecuta UPDATE. Sin esto solo muestra auditoria.")
    parser.add_argument(
        "--require-source-time",
        action="store_true",
        help="Solo actualiza filas cuyo tiempo exista como TA en archivo_plano_detalle.",
    )
    args = parser.parse_args()

    rows = cargar_destino(args)
    tiempos = cargar_tiempos_archivo_plano(rows)
    updates, stats = armar_actualizaciones(rows, tiempos, args)

    sample = updates[:20]
    if args.apply and updates:
        aplicar_actualizaciones(updates)

    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "source": "archivo_plano_cabecera + archivo_plano_detalle",
        "target": "ap_auditoria_doctor_detalle",
        "rows_read": len(rows),
        "source_time_keys": len(tiempos),
        "updates": len(updates),
        "stats": stats,
        "sample": sample,
    }, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
