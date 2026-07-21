export type NivelRiesgo = "Alto" | "Medio" | "Bajo";

export type EstadoCaso =
  | "pendiente"
  | "contactado"
  | "en_seguimiento"
  | "cerrado";

export type Severidad = "critico" | "alto" | "medio" | "protector";

export interface Estudiante {
  codigo: string;
  nombre_completo: string | null;
  carrera: string | null;
  ciclo: number | null;
  correo: string | null;
}

export interface FactorPrincipal {
  mensaje: string;
  severidad: Severidad;
}

export interface Factor {
  codigo: string;
  mensaje: string;
  severidad: Severidad;
  valor: number | null;
  sugerencia: string | null;
}

export interface CasoResumen {
  prediction_id: number;
  student_id: string;
  estado_predicho: string;
  nivel_riesgo: NivelRiesgo;
  confianza: number;
  created_at: string;
  estado_caso: EstadoCaso;
  responsable: string | null;
  nota: string | null;
  factor_principal: FactorPrincipal | null;
  total_factores_riesgo: number;
  estudiante: Estudiante;
}

export interface PaginaEstudiantes {
  total: number;
  pagina: number;
  por_pagina: number;
  paginas: number;
  estudiantes: CasoResumen[];
}

export interface Explicacion {
  prediction_id: number;
  student_id: string;
  estado_predicho: string;
  nivel_riesgo: NivelRiesgo;
  confianza: number;
  resumen: string;
  total_factores_riesgo: number;
  factores_riesgo: Factor[];
  factores_protectores: Factor[];
  nota: string;
  estudiante: Estudiante;
}

export interface CasoActualizado {
  id: number;
  prediction_id: number;
  student_id: string;
  estado: EstadoCaso;
  responsable: string | null;
  nota: string | null;
  updated_at: string;
}

export interface FiltrosBandeja {
  nivel_riesgo?: NivelRiesgo;
  estado?: EstadoCaso;
  buscar?: string;
  pagina?: number;
  orden?: "reciente" | "confianza" | "riesgo";
}
