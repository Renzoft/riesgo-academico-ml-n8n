const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

export class ErrorApi extends Error {
  estado: number;

  constructor(estado: number, mensaje: string) {
    super(mensaje);
    this.estado = estado;
  }
}

async function pedir<T>(ruta: string, opciones: RequestInit = {}): Promise<T> {
  const cabeceras: Record<string, string> = {
    "Content-Type": "application/json",
    ...((opciones.headers as Record<string, string>) ?? {}),
  };
  if (API_KEY) cabeceras["X-API-Key"] = API_KEY;

  const respuesta = await fetch(`${BASE}${ruta}`, {
    ...opciones,
    headers: cabeceras,
  });

  if (!respuesta.ok) {
    // La API devuelve el motivo en "detail"; se conserva para mostrarlo.
    let detalle = `Error ${respuesta.status}`;
    try {
      const cuerpo = await respuesta.json();
      if (cuerpo?.detail) detalle = String(cuerpo.detail);
    } catch {
      /* respuesta sin cuerpo JSON */
    }
    throw new ErrorApi(respuesta.status, detalle);
  }

  return respuesta.json() as Promise<T>;
}

export const api = {
  get: <T>(ruta: string) => pedir<T>(ruta),
  patch: <T>(ruta: string, cuerpo: unknown) =>
    pedir<T>(ruta, { method: "PATCH", body: JSON.stringify(cuerpo) }),
};
