import { Link, useSearchParams } from "react-router-dom";
import { useEstudiantes, useResumen } from "../api/consultas";
import type { CasoResumen, EstadoCaso, NivelRiesgo } from "../api/tipos";
import {
  certezaEnPalabras,
  EstadoVacio,
  InsigniaEstado,
  InsigniaRiesgo,
  MensajeError,
} from "../componentes/Indicadores";

function BarraResumen() {
  const { data } = useResumen();
  if (!data) return null;

  const riesgo = data.por_nivel_riesgo;
  return (
    <div className="resumen">
      <div className="resumen-dato">
        <span className="resumen-cifra">{data.total_estudiantes}</span>
        <span className="resumen-etiqueta">Estudiantes</span>
      </div>
      <div className="resumen-dato">
        <span className="resumen-cifra alto">{riesgo.Alto ?? 0}</span>
        <span className="resumen-etiqueta">Riesgo alto</span>
      </div>
      <div className="resumen-dato">
        <span className="resumen-cifra">{riesgo.Medio ?? 0}</span>
        <span className="resumen-etiqueta">Riesgo medio</span>
      </div>
      <div className="resumen-dato">
        <span className="resumen-cifra">{riesgo.Bajo ?? 0}</span>
        <span className="resumen-etiqueta">Riesgo bajo</span>
      </div>
      <div className="resumen-dato">
        <span className="resumen-cifra">{data.pendientes_de_atencion}</span>
        <span className="resumen-etiqueta">Sin atender</span>
      </div>
    </div>
  );
}

function Fila({ caso }: { caso: CasoResumen }) {
  const { estudiante } = caso;
  const nombre = estudiante.nombre_completo ?? caso.student_id;
  const enDirectorio = estudiante.nombre_completo !== null;

  return (
    <Link to={`/caso/${caso.prediction_id}`} className="caso">
      <div className="caso-linea">
        <InsigniaRiesgo nivel={caso.nivel_riesgo} />
        <span className="caso-nombre">{nombre}</span>
        {enDirectorio && (
          <span className="caso-meta">
            {estudiante.codigo}
            {estudiante.carrera && ` · ${estudiante.carrera}`}
            {estudiante.ciclo && ` · ciclo ${estudiante.ciclo}`}
          </span>
        )}
      </div>

      <div className="caso-motivo">
        {caso.factor_principal
          ? caso.factor_principal.mensaje
          : "Sin factores de riesgo identificados en las variables observadas."}
      </div>

      <div className="caso-pie">
        <span>{certezaEnPalabras(caso.confianza)}</span>
        {caso.total_factores_riesgo > 1 && (
          <>
            <span className="separador">|</span>
            <span>{caso.total_factores_riesgo} factores</span>
          </>
        )}
        <span className="separador">|</span>
        <InsigniaEstado estado={caso.estado_caso} />
      </div>
    </Link>
  );
}

export default function Bandeja() {
  // Los filtros viven en la URL: se pueden compartir y sobreviven al refresco.
  const [parametros, setParametros] = useSearchParams();

  const filtros = {
    nivel_riesgo: (parametros.get("riesgo") as NivelRiesgo) || undefined,
    estado: (parametros.get("estado") as EstadoCaso) || "pendiente",
    buscar: parametros.get("buscar") || undefined,
    pagina: Number(parametros.get("pagina") ?? 1),
    orden: "riesgo" as const,
  };

  const cambiar = (clave: string, valor: string) => {
    const siguiente = new URLSearchParams(parametros);
    if (valor) siguiente.set(clave, valor);
    else siguiente.delete(clave);
    if (clave !== "pagina") siguiente.delete("pagina");
    setParametros(siguiente);
  };

  const { data, isPending, isError, error } = useEstudiantes(filtros);

  return (
    <div className="contenedor">
      <BarraResumen />

      <div className="filtros">
        <select
          value={parametros.get("riesgo") ?? ""}
          onChange={(e) => cambiar("riesgo", e.target.value)}
          aria-label="Filtrar por nivel de riesgo"
        >
          <option value="">Todos los niveles</option>
          <option value="Alto">Riesgo alto</option>
          <option value="Medio">Riesgo medio</option>
          <option value="Bajo">Riesgo bajo</option>
        </select>

        <select
          value={parametros.get("estado") ?? "pendiente"}
          onChange={(e) => cambiar("estado", e.target.value)}
          aria-label="Filtrar por estado del caso"
        >
          <option value="pendiente">Sin atender</option>
          <option value="contactado">Contactados</option>
          <option value="en_seguimiento">En seguimiento</option>
          <option value="cerrado">Cerrados</option>
          <option value="">Todos los estados</option>
        </select>

        <input
          type="search"
          placeholder="Buscar por código de estudiante"
          defaultValue={parametros.get("buscar") ?? ""}
          onChange={(e) => cambiar("buscar", e.target.value)}
          aria-label="Buscar estudiante"
        />
      </div>

      {isError && <MensajeError error={error} />}

      {isPending && (
        <EstadoVacio titulo="Cargando casos" detalle="Un momento." />
      )}

      {data && data.estudiantes.length === 0 && (
        <EstadoVacio
          titulo="No hay casos con estos filtros"
          detalle="Prueba con otro nivel de riesgo o cambia el estado."
        />
      )}

      {data && data.estudiantes.length > 0 && (
        <>
          <p className="conteo">
            {data.total} caso{data.total === 1 ? "" : "s"}
            {filtros.estado === "pendiente" && " sin atender"}
          </p>

          {data.estudiantes.map((caso) => (
            <Fila key={caso.prediction_id} caso={caso} />
          ))}

          {data.paginas > 1 && (
            <div className="paginacion">
              <button
                disabled={filtros.pagina <= 1}
                onClick={() => cambiar("pagina", String(filtros.pagina - 1))}
              >
                Anterior
              </button>
              <span>
                Página {data.pagina} de {data.paginas}
              </span>
              <button
                disabled={filtros.pagina >= data.paginas}
                onClick={() => cambiar("pagina", String(filtros.pagina + 1))}
              >
                Siguiente
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
