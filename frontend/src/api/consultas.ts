import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  CasoActualizado,
  EstadoCaso,
  Explicacion,
  FiltrosBandeja,
  PaginaEstudiantes,
} from "./tipos";

function construirBusqueda(filtros: FiltrosBandeja): string {
  const parametros = new URLSearchParams();
  if (filtros.nivel_riesgo) parametros.set("nivel_riesgo", filtros.nivel_riesgo);
  if (filtros.estado) parametros.set("estado", filtros.estado);
  if (filtros.buscar) parametros.set("buscar", filtros.buscar);
  parametros.set("pagina", String(filtros.pagina ?? 1));
  parametros.set("orden", filtros.orden ?? "riesgo");
  parametros.set("por_pagina", "20");
  return parametros.toString();
}

export function useEstudiantes(filtros: FiltrosBandeja) {
  return useQuery({
    queryKey: ["estudiantes", filtros],
    queryFn: () =>
      api.get<PaginaEstudiantes>(
        `/panel/estudiantes?${construirBusqueda(filtros)}`,
      ),
    placeholderData: (anterior) => anterior,
  });
}

export interface Resumen {
  total_estudiantes: number;
  confianza_media: number | null;
  por_nivel_riesgo: Record<string, number>;
  por_estado_caso: Record<string, number>;
  pendientes_de_atencion: number;
  cobertura_seguimiento: number;
}

export function useResumen() {
  return useQuery({
    queryKey: ["resumen"],
    queryFn: () => api.get<Resumen>("/panel/resumen"),
  });
}

export function useExplicacion(predictionId: number | undefined) {
  return useQuery({
    queryKey: ["explicacion", predictionId],
    queryFn: () =>
      api.get<Explicacion>(
        `/panel/predicciones/${predictionId}/explicacion`,
      ),
    enabled: predictionId !== undefined,
  });
}

export function useActualizarCaso(predictionId: number) {
  const cliente = useQueryClient();
  return useMutation({
    mutationFn: (cambio: {
      estado: EstadoCaso;
      responsable?: string;
      nota?: string;
    }) => api.patch<CasoActualizado>(`/panel/casos/${predictionId}`, cambio),
    onSuccess: () => {
      // La bandeja filtra por estado, asi que debe refrescarse.
      cliente.invalidateQueries({ queryKey: ["estudiantes"] });
      cliente.invalidateQueries({ queryKey: ["resumen"] });
      cliente.invalidateQueries({ queryKey: ["explicacion", predictionId] });
    },
  });
}
