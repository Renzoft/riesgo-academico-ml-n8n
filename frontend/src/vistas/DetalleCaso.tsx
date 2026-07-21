import { Link, useParams } from "react-router-dom";
import { useActualizarCaso, useExplicacion } from "../api/consultas";
import type { EstadoCaso } from "../api/tipos";
import {
  certezaEnPalabras,
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
        <div className="caso-linea">
          <InsigniaRiesgo nivel={data.nivel_riesgo} />
          <h2 style={{ margin: 0, fontSize: 19, textTransform: "none", color: "var(--texto)", letterSpacing: 0 }}>
            {nombre}
          </h2>
        </div>
        <div className="caso-meta" style={{ marginTop: 6 }}>
          {estudiante.codigo}
          {estudiante.carrera && ` · ${estudiante.carrera}`}
          {estudiante.ciclo && ` · ciclo ${estudiante.ciclo}`}
        </div>
        {estudiante.correo && (
          <div className="caso-meta">{estudiante.correo}</div>
        )}
        {estudiante.nombre_completo === null && (
          <div className="caso-meta" style={{ marginTop: 8 }}>
            Este código no figura en el directorio académico.
          </div>
        )}
      </div>

      <div className="panel">
        <h2>Predicción del modelo</h2>
        <p style={{ margin: "0 0 14px" }}>
          Estado previsto: <strong>{data.estado_predicho}</strong> ·{" "}
          {certezaEnPalabras(data.confianza)} (
          {(data.confianza * 100).toFixed(1)}%)
        </p>
      </div>

      <div className="panel">
        <h2>Por qué aparece este caso</h2>

        {data.factores_riesgo.length === 0 && (
          <p style={{ margin: 0 }}>
            No se identificaron factores de riesgo entre las variables
            observadas.
          </p>
        )}

        {data.factores_riesgo.map((factor) => (
          <LineaFactor key={factor.codigo} factor={factor} />
        ))}

        {data.factores_protectores.length > 0 && (
          <>
            <h2 style={{ marginTop: 20 }}>Factores protectores</h2>
            {data.factores_protectores.map((factor) => (
              <LineaFactor key={factor.codigo} factor={factor} />
            ))}
          </>
        )}

        <p className="nota-pie">{data.nota}</p>
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
          <p className="caso-meta" style={{ marginTop: 12 }}>
            Caso actualizado a{" "}
            <strong>{actualizar.data.estado.replace("_", " ")}</strong>.
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
