import type { EstadoCaso, Factor, NivelRiesgo, Severidad } from "../api/tipos";

export function InsigniaRiesgo({ nivel }: { nivel: NivelRiesgo }) {
  return (
    <span className={`insignia ${nivel.toLowerCase()}`}>
      Riesgo {nivel.toLowerCase()}
    </span>
  );
}

/**
 * La confianza se muestra en palabras. Un tutor no interpreta "0.673", y el
 * corte de 0.60 es el mismo umbral que usa n8n para derivar a revision.
 */
export function certezaEnPalabras(confianza: number): string {
  if (confianza >= 0.8) return "certeza alta";
  if (confianza >= 0.6) return "certeza media";
  return "requiere verificación";
}

const ETIQUETA_ESTADO: Record<EstadoCaso, string> = {
  pendiente: "Pendiente",
  contactado: "Contactado",
  en_seguimiento: "En seguimiento",
  cerrado: "Cerrado",
};

export function InsigniaEstado({ estado }: { estado: EstadoCaso }) {
  return <span className="insignia estado">{ETIQUETA_ESTADO[estado]}</span>;
}

const ETIQUETA_SEVERIDAD: Record<Severidad, string> = {
  critico: "Crítico",
  alto: "Alto",
  medio: "Medio",
  protector: "Protector",
};

export function LineaFactor({ factor }: { factor: Factor }) {
  return (
    <div className="factor">
      <div className={`factor-marca ${factor.severidad}`} />
      <div>
        <div className={`factor-etiqueta ${factor.severidad}`}>
          {ETIQUETA_SEVERIDAD[factor.severidad]}
        </div>
        <div>{factor.mensaje}</div>
        {factor.sugerencia && (
          <div className="factor-sugerencia">{factor.sugerencia}</div>
        )}
      </div>
    </div>
  );
}

export function EstadoVacio({
  titulo,
  detalle,
}: {
  titulo: string;
  detalle: string;
}) {
  return (
    <div className="vacio">
      <strong>{titulo}</strong>
      {detalle}
    </div>
  );
}

export function MensajeError({ error }: { error: unknown }) {
  const texto =
    error instanceof Error ? error.message : "Ocurrió un error inesperado.";
  return (
    <div className="error">
      No se pudo cargar la información. {texto}
    </div>
  );
}
