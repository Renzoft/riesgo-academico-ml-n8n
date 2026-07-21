import { Link, useParams } from "react-router-dom";
import { useActualizarCaso, useExplicacion } from "../api/consultas";
import type { EstadoCaso } from "../api/tipos";
import {
  certezaEnPalabras,
  claseEnEspanol,
  EstadoVacio,
  InsigniaRiesgo,
  LineaFactor,
  MensajeError,
} from "../componentes/Indicadores";

const TRANSICIONES: { estado: EstadoCaso; etiqueta: string }[] = [
  { estado: "contactado", etiqueta: "Marcar como contactado" },
  { estado: "en_seguimiento", etiqueta: "Poner en seguimiento" },
  { estado: "cerrado", etiqueta: "Cerrar caso" },
];

const ETIQUETA_ESTADO: Record<EstadoCaso, string> = {
  pendiente: "sin atender",
  contactado: "contactado",
  en_seguimiento: "en seguimiento",
  cerrado: "cerrado",
};

export default function DetalleCaso() {
  const { id } = useParams<{ id: string }>();
  const predictionId = Number(id);

  const { data, isPending, isError, error } = useExplicacion(predictionId);
  const actualizar = useActualizarCaso(predictionId);

  if (isError) {
    return (
      <div className="contenedor">
        <Link to="/" className="enlace-volver">
          ← Volver a la bandeja
        </Link>
        <MensajeError error={error} />
      </div>
    );
  }

  if (isPending || !data) {
    return (
      <div className="contenedor">
        <EstadoVacio titulo="Cargando el caso" detalle="Un momento." />
      </div>
    );
  }

  const { estudiante } = data;
  const nombre = estudiante.nombre_completo ?? data.student_id;

  return (
    <div className="contenedor">
      <Link to="/" className="enlace-volver">
        ← Volver a la bandeja
      </Link>

      <div className="panel">
        <h1 className="ficha-nombre">{nombre}</h1>
        <div className="ficha-datos">
          {estudiante.codigo}
          {estudiante.carrera && ` · ${estudiante.carrera}`}
          {estudiante.ciclo && ` · ciclo ${estudiante.ciclo}`}
          {estudiante.correo && ` · ${estudiante.correo}`}
        </div>
        {estudiante.nombre_completo === null && (
          <div className="ficha-datos">
            Este código no figura en el directorio académico.
          </div>
        )}
      </div>

      <div className="panel">
        <h2>Evaluación del modelo</h2>
        <div className="caso-linea" style={{ marginBottom: 8 }}>
          <InsigniaRiesgo nivel={data.nivel_riesgo} />
          <span className="caso-meta">
            {certezaEnPalabras(data.confianza)} ·{" "}
            {(data.confianza * 100).toFixed(1)}%
          </span>
        </div>
        <p className="dato-destacado">
          Desenlace previsto:{" "}
          <strong>{claseEnEspanol(data.estado_predicho)}</strong>
        </p>
      </div>

      <div className="panel">
        <h2>Factores observados</h2>

        {data.factores_riesgo.length === 0 && (
          <p style={{ margin: 0, color: "var(--texto-medio)" }}>
            No se identificaron factores de riesgo entre las variables
            observadas.
          </p>
        )}

        {data.factores_riesgo.map((factor) => (
          <LineaFactor key={factor.codigo} factor={factor} />
        ))}

        {data.factores_protectores.length > 0 && (
          <>
            <h2 style={{ marginTop: 22 }}>Factores protectores</h2>
            {data.factores_protectores.map((factor) => (
              <LineaFactor key={factor.codigo} factor={factor} />
            ))}
          </>
        )}
      </div>

      <div className="panel">
        <h2>Gestión del caso</h2>
        <div className="acciones">
          {TRANSICIONES.map(({ estado, etiqueta }) => (
            <button
              key={estado}
              disabled={actualizar.isPending}
              onClick={() =>
                actualizar.mutate({ estado, responsable: "Tutor" })
              }
            >
              {etiqueta}
            </button>
          ))}
        </div>

        {actualizar.isSuccess && (
          <p className="aviso-ok">
            Caso registrado como{" "}
            <strong>{ETIQUETA_ESTADO[actualizar.data.estado]}</strong>.
          </p>
        )}
        {actualizar.isError && (
          <div style={{ marginTop: 12 }}>
            <MensajeError error={actualizar.error} />
          </div>
        )}
      </div>
    </div>
  );
}
