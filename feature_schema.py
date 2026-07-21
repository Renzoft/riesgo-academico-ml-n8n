from pydantic import BaseModel, ConfigDict, Field


# Relacion unica entre el contrato publico de la API y las columnas usadas
# durante el entrenamiento. El orden de este diccionario es el orden exacto
# que esperan el scaler y el modelo actuales.
FEATURE_FIELD_MAP = {
    "marital_status": "Marital status",
    "application_mode": "Application mode",
    "application_order": "Application order",
    "course": "Course",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "previous_qualification": "Previous qualification",
    "nationality": "Nacionality",
    "mothers_qualification": "Mother's qualification",
    "fathers_qualification": "Father's qualification",
    "mothers_occupation": "Mother's occupation",
    "fathers_occupation": "Father's occupation",
    "displaced": "Displaced",
    "educational_special_needs": "Educational special needs",
    "debtor": "Debtor",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "gender": "Gender",
    "scholarship_holder": "Scholarship holder",
    "age_at_enrollment": "Age at enrollment",
    "international": "International",
    "curricular_units_1st_sem_credited": (
        "Curricular units 1st sem (credited)"
    ),
    "curricular_units_1st_sem_enrolled": (
        "Curricular units 1st sem (enrolled)"
    ),
    "curricular_units_1st_sem_evaluations": (
        "Curricular units 1st sem (evaluations)"
    ),
    "curricular_units_1st_sem_approved": (
        "Curricular units 1st sem (approved)"
    ),
    "curricular_units_1st_sem_grade": (
        "Curricular units 1st sem (grade)"
    ),
    "curricular_units_1st_sem_without_evaluations": (
        "Curricular units 1st sem (without evaluations)"
    ),
    "curricular_units_2nd_sem_credited": (
        "Curricular units 2nd sem (credited)"
    ),
    "curricular_units_2nd_sem_enrolled": (
        "Curricular units 2nd sem (enrolled)"
    ),
    "curricular_units_2nd_sem_evaluations": (
        "Curricular units 2nd sem (evaluations)"
    ),
    "curricular_units_2nd_sem_approved": (
        "Curricular units 2nd sem (approved)"
    ),
    "curricular_units_2nd_sem_grade": (
        "Curricular units 2nd sem (grade)"
    ),
    "curricular_units_2nd_sem_without_evaluations": (
        "Curricular units 2nd sem (without evaluations)"
    ),
    "unemployment_rate": "Unemployment rate",
    "inflation_rate": "Inflation rate",
    "gdp": "GDP",
}

CATEGORICAL_API_FIELDS = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nationality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]

BINARY_API_FIELDS = [
    "daytime_evening_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

NUMERIC_API_FIELDS = [
    name
    for name in FEATURE_FIELD_MAP
    if name not in CATEGORICAL_API_FIELDS + BINARY_API_FIELDS
]


class StudentFeatures(BaseModel):
    """Caracteristicas nombradas que recibe la API publica."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    marital_status: int = Field(ge=1)
    application_mode: int = Field(ge=1)
    application_order: int = Field(ge=0)
    course: int = Field(ge=1)
    daytime_evening_attendance: int = Field(ge=0, le=1)
    previous_qualification: int = Field(ge=1)
    nationality: int = Field(ge=1)
    mothers_qualification: int = Field(ge=1)
    fathers_qualification: int = Field(ge=1)
    mothers_occupation: int = Field(ge=0)
    fathers_occupation: int = Field(ge=0)
    displaced: int = Field(ge=0, le=1)
    educational_special_needs: int = Field(ge=0, le=1)
    debtor: int = Field(ge=0, le=1)
    tuition_fees_up_to_date: int = Field(ge=0, le=1)
    gender: int = Field(ge=0, le=1)
    scholarship_holder: int = Field(ge=0, le=1)
    age_at_enrollment: int = Field(ge=15, le=100)
    international: int = Field(ge=0, le=1)
    curricular_units_1st_sem_credited: int = Field(ge=0)
    curricular_units_1st_sem_enrolled: int = Field(ge=0)
    curricular_units_1st_sem_evaluations: int = Field(ge=0)
    curricular_units_1st_sem_approved: int = Field(ge=0)
    curricular_units_1st_sem_grade: float = Field(ge=0, le=20)
    curricular_units_1st_sem_without_evaluations: int = Field(ge=0)
    curricular_units_2nd_sem_credited: int = Field(ge=0)
    curricular_units_2nd_sem_enrolled: int = Field(ge=0)
    curricular_units_2nd_sem_evaluations: int = Field(ge=0)
    curricular_units_2nd_sem_approved: int = Field(ge=0)
    curricular_units_2nd_sem_grade: float = Field(ge=0, le=20)
    curricular_units_2nd_sem_without_evaluations: int = Field(ge=0)
    unemployment_rate: float
    inflation_rate: float
    gdp: float

    def to_model_vector(self, model_feature_names: list[str]) -> list[float]:
        """Construye el vector segun el orden declarado por el modelo."""
        values_by_model_name = {
            model_name: float(getattr(self, api_name))
            for api_name, model_name in FEATURE_FIELD_MAP.items()
        }

        missing = [
            name for name in model_feature_names
            if name not in values_by_model_name
        ]
        additional = [
            name for name in values_by_model_name
            if name not in model_feature_names
        ]
        if missing or additional:
            raise ValueError(
                "El esquema de entrada no coincide con el modelo. "
                f"Faltantes: {missing}. Adicionales: {additional}."
            )

        return [values_by_model_name[name] for name in model_feature_names]
