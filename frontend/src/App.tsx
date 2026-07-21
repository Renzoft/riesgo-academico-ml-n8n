import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import Bandeja from "./vistas/Bandeja";
import DetalleCaso from "./vistas/DetalleCaso";

const cliente = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={cliente}>
      <BrowserRouter>
        <header className="cabecera">
          <div className="contenedor">
            <Link to="/" style={{ textDecoration: "none", color: "inherit" }}>
              <h1>Seguimiento de riesgo académico</h1>
            </Link>
            <p>Detección temprana de abandono · UNMSM</p>
          </div>
        </header>

        <Routes>
          <Route path="/" element={<Bandeja />} />
          <Route path="/caso/:id" element={<DetalleCaso />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
