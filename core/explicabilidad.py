"""Traduce las variables de un estudiante a factores legibles por un tutor.

Importante: estos factores NO son una atribucion del modelo. Son indicadores
de contexto derivados de las variables que la literatura y el propio dataset
senalan como asociadas al abandono. Sirven para que el tutor sepa de que
hablar con el estudiante, no para explicar el calculo interno de la red.

La atribucion real por prediccion (SHAP o importancia por permutacion) queda
como trabajo posterior.
"""

CRITICO = "critico"
ALTO = "alto"
MEDIO = "medio"
PROTECTOR = "protector"

# Severidades ordenadas para poder priorizar la lista que ve el tutor.
ORDEN_SEVERIDAD = {CRITICO: 0, ALTO: 1, MEDIO: 2, PROTECTOR: 3}


def _valor(features: dict, nombre: str, defecto: float = 0.0) -> float:
    try:
        return float(features.get(nombre, defecto))
    except (TypeError, ValueError):
        return defecto


def _factor(codigo, mensaje, severidad, valor=None, sugerencia=None) -> dict:
    return {
        "codigo": codigo,
        "mensaje": mensaje,
        "severidad": severidad,
        "valor": valor,
        "sugerencia": sugerencia,
    }


def _factores_academicos(features: dict) -> list[dict]:
    factores = []

    aprobados_1 = _valor(features, "Curricular units 1st sem (approved)")
    aprobados_2 = _valor(features, "Curricular units 2nd sem (approved)")
    matriculados_1 = _valor(features, "Curricular units 1st sem (enrolled)")
    matriculados_2 = _valor(features, "Curricular units 2nd sem (enrolled)")
    nota_1 = _valor(features, "Curricular units 1st sem (grade)")
    nota_2 = _valor(features, "Curricular units 2nd sem (grade)")

    # Desvinculacion total: la senal mas fuerte del dataset. Se evalua antes
    # que cualquier otra regla academica porque las vuelve irrelevantes.
    if matriculados_1 == 0 and matriculados_2 == 0:
        factores.append(_factor(
            "sin_matricula",
            "No se matriculó en ningún curso durante el primer año.",
            CRITICO,
            valor=0,
            sugerencia=(
                "Verificar si continúa vinculado a la institución antes de "
                "cualquier otra gestión."
            ),
        ))
        return factores

    if matriculados_1 > 0 and matriculados_2 == 0:
        factores.append(_factor(
            "abandono_segundo_semestre",
            "No se matriculó en el segundo semestre pese a haber cursado el "
            f"primero (matriculó {int(matriculados_1)} cursos).",
            CRITICO,
            valor=0,
            sugerencia="Contactar para conocer el motivo de la interrupción.",
        ))
        return factores

    if matriculados_2 > 0 and aprobados_2 == 0:
        factores.append(_factor(
            "sin_aprobados_2do",
            f"No aprobó ningún curso en el segundo semestre "
            f"(matriculó {int(matriculados_2)}).",
            CRITICO,
            valor=aprobados_2,
            sugerencia="Contactar de inmediato y evaluar causas académicas.",
        ))
    elif matriculados_2 > 0:
        ratio = aprobados_2 / matriculados_2
        if ratio < 0.5:
            factores.append(_factor(
                "bajo_ratio_aprobacion",
                f"Aprobó {int(aprobados_2)} de {int(matriculados_2)} cursos "
                f"del segundo semestre ({ratio:.0%}).",
                ALTO,
                valor=round(ratio, 3),
                sugerencia="Revisar carga académica y necesidad de tutoría.",
            ))

    if nota_2 > 0 and nota_2 < 10:
        factores.append(_factor(
            "nota_baja_2do",
            f"Promedio del segundo semestre en {nota_2:.1f}, "
            "por debajo del mínimo aprobatorio.",
            ALTO,
            valor=round(nota_2, 2),
            sugerencia="Derivar a apoyo académico.",
        ))

    # La tendencia entre semestres suele importar mas que el valor absoluto.
    if aprobados_1 > 0 and aprobados_2 < aprobados_1:
        factores.append(_factor(
            "tendencia_negativa",
            f"Su rendimiento empeoró: aprobó {int(aprobados_1)} cursos en el "
            f"primer semestre y {int(aprobados_2)} en el segundo.",
            MEDIO,
            valor=aprobados_2 - aprobados_1,
            sugerencia="Indagar qué cambió entre ambos periodos.",
        ))

    if nota_1 > 0 and nota_2 > 0 and (nota_1 - nota_2) >= 2:
        factores.append(_factor(
            "caida_promedio",
            f"Su promedio cayó de {nota_1:.1f} a {nota_2:.1f} entre semestres.",
            MEDIO,
            valor=round(nota_2 - nota_1, 2),
        ))

    return factores


def _factores_economicos(features: dict) -> list[dict]:
    factores = []

    if _valor(features, "Debtor") == 1:
        factores.append(_factor(
            "deudor",
            "Registra deuda pendiente con la institución.",
            ALTO,
            valor=1,
            sugerencia="Derivar a bienestar o a la oficina de becas.",
        ))

    if _valor(features, "Tuition fees up to date") == 0:
        factores.append(_factor(
            "matricula_impaga",
            "No está al día con el pago de la matrícula.",
            ALTO,
            valor=0,
            sugerencia="Verificar situación económica antes de intervenir.",
        ))

    if _valor(features, "Scholarship holder") == 1:
        factores.append(_factor(
            "becario",
            "Cuenta con beca vigente.",
            PROTECTOR,
            valor=1,
        ))

    return factores


def _factores_contexto(features: dict) -> list[dict]:
    factores = []

    edad = _valor(features, "Age at enrollment")
    if edad >= 25:
        factores.append(_factor(
            "ingreso_tardio",
            f"Ingresó a los {int(edad)} años, por encima de la edad habitual.",
            MEDIO,
            valor=edad,
            sugerencia="Suele implicar carga laboral o familiar en paralelo.",
        ))

    if _valor(features, "Displaced") == 1:
        factores.append(_factor(
            "desplazado",
            "Está fuera de su lugar de residencia habitual.",
            MEDIO,
            valor=1,
        ))

    if _valor(features, "Educational special needs") == 1:
        factores.append(_factor(
            "necesidades_especiales",
            "Registra necesidades educativas especiales.",
            MEDIO,
            valor=1,
            sugerencia="Verificar que reciba los apoyos previstos.",
        ))

    if _valor(features, "Daytime/evening attendance") == 0:
        factores.append(_factor(
            "turno_noche",
            "Asiste en turno nocturno.",
            MEDIO,
            valor=0,
        ))

    return factores


def explicar_estudiante(features: dict) -> dict:
    """Devuelve los factores de riesgo y protectores de un estudiante.

    ``features`` es el diccionario {nombre_de_variable: valor} tal como se
    guarda en la columna input_features de la tabla predictions.
    """
    factores = (
        _factores_academicos(features)
        + _factores_economicos(features)
        + _factores_contexto(features)
    )
    factores.sort(key=lambda f: ORDEN_SEVERIDAD[f["severidad"]])

    de_riesgo = [f for f in factores if f["severidad"] != PROTECTOR]
    protectores = [f for f in factores if f["severidad"] == PROTECTOR]

    if not de_riesgo:
        resumen = (
            "No se identificaron factores de riesgo entre las variables "
            "observadas."
        )
    else:
        resumen = de_riesgo[0]["mensaje"]

    return {
        "resumen": resumen,
        "total_factores_riesgo": len(de_riesgo),
        "factores_riesgo": de_riesgo,
        "factores_protectores": protectores,
        "nota": (
            "Indicadores de contexto derivados de las variables del "
            "estudiante. No constituyen una atribución del modelo."
        ),
    }
