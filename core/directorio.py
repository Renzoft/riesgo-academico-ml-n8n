"""Directorio de estudiantes: frontera de integracion con el sistema academico.

En una implantacion real, los datos personales del estudiante (nombre,
carrera, ciclo, correo) pertenecen al sistema academico de la institucion,
que es el dueno de esa informacion. Este sistema de riesgo NO los almacena
junto a las predicciones: guarda unicamente el codigo del estudiante y
resuelve el resto en el momento de mostrarlo.

Aqui ese sistema academico se simula con una tabla local cargada desde
data/directorio_estudiantes.csv, con estudiantes ficticios. Para conectar
el sistema real basta con reemplazar el cuerpo de ``buscar_por_codigos``
por la llamada correspondiente a su API, sin tocar nada mas.
"""

import csv
import sqlite3
from pathlib import Path

DIRECTORIO_CSV = Path("data/directorio_estudiantes.csv")


def asegurar_tabla(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS student_directory (
            codigo TEXT PRIMARY KEY,
            nombres TEXT NOT NULL,
            apellidos TEXT NOT NULL,
            carrera TEXT,
            ciclo INTEGER,
            correo TEXT
        )
        """
    )
    connection.commit()


def cargar_directorio(
    connection: sqlite3.Connection,
    csv_path: Path = DIRECTORIO_CSV,
) -> int:
    """Carga el directorio simulado. Idempotente: repetirlo no duplica."""
    asegurar_tabla(connection)

    if not csv_path.exists():
        return 0

    with csv_path.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    connection.executemany(
        """
        INSERT INTO student_directory
            (codigo, nombres, apellidos, carrera, ciclo, correo)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(codigo) DO UPDATE SET
            nombres = excluded.nombres,
            apellidos = excluded.apellidos,
            carrera = excluded.carrera,
            ciclo = excluded.ciclo,
            correo = excluded.correo
        """,
        [
            (
                fila["codigo"],
                fila["nombres"],
                fila["apellidos"],
                fila.get("carrera"),
                int(fila["ciclo"]) if fila.get("ciclo") else None,
                fila.get("correo"),
            )
            for fila in filas
        ],
    )
    connection.commit()
    return len(filas)


def buscar_por_codigos(
    connection: sqlite3.Connection,
    codigos: list[str],
) -> dict[str, dict]:
    """Resuelve los datos de varios estudiantes en una sola consulta.

    Este es el punto de integracion: al conectar el sistema academico real,
    se sustituye esta consulta por la llamada a su servicio manteniendo el
    mismo contrato de entrada y salida.
    """
    if not codigos:
        return {}

    marcadores = ",".join("?" * len(codigos))
    filas = connection.execute(
        f"SELECT * FROM student_directory WHERE codigo IN ({marcadores})",
        codigos,
    ).fetchall()

    return {fila["codigo"]: dict(fila) for fila in filas}


def enriquecer(registros: list[dict], directorio: dict[str, dict]) -> None:
    """Agrega los datos del estudiante a cada registro, en el sitio.

    Si el codigo no existe en el directorio, se conserva el identificador
    tal cual para que el panel siga siendo utilizable.
    """
    for registro in registros:
        datos = directorio.get(registro.get("student_id"))
        if datos:
            registro["estudiante"] = {
                "codigo": datos["codigo"],
                "nombre_completo": (
                    f"{datos['nombres']} {datos['apellidos']}"
                ),
                "carrera": datos["carrera"],
                "ciclo": datos["ciclo"],
                "correo": datos["correo"],
            }
        else:
            registro["estudiante"] = {
                "codigo": registro.get("student_id"),
                "nombre_completo": None,
                "carrera": None,
                "ciclo": None,
                "correo": None,
            }
