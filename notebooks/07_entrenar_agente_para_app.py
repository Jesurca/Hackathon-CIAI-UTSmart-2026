# Entrena el agente por refuerzo con los datos reales del kit y guarda el .npz para la app.
# Usa el mismo flujo de la app (agente_rl), así el archivo queda "casado" con estos datos.
# Uso: python notebooks/07_entrenar_agente_para_app.py [episodios]
import sys
import time
from pathlib import Path

import pandas as pd

# Encuentra la raíz del repo (sube carpetas hasta hallar "datos").
RAIZ = Path.cwd()
while not (RAIZ / "datos").exists() and RAIZ.parent != RAIZ:
    RAIZ = RAIZ.parent

# La app vive en su propia carpeta: se importa su paquete sin copiar código.
sys.path.insert(0, str(RAIZ / "app/sismo_app_F/sismo_app"))
from agente_rl import servicio                     # noqa: E402
from agente_rl.flujo import CFG_BASE, procesar     # noqa: E402

KIT = RAIZ / "datos/para_app"
SALIDA = KIT / "agente_sismico.npz"
EPISODIOS = int(sys.argv[1]) if len(sys.argv) > 1 else 300

cat = pd.read_csv(KIT / "catalogo_sgc.csv")
ofi = pd.read_csv(KIT / "fuente_oficial.csv")
sup = pd.read_csv(KIT / "supervisado_resultados.csv")
clu = pd.read_csv(KIT / "clusters_resultados.csv")

cfg = {**CFG_BASE, "entorno": dict(CFG_BASE["entorno"])}
R = procesar(cat, ofi, cfg, sup, clu)
rep = R["reporte"]
print(f"[panel] {R['panel'].shape} | cortes {R['cfg']['fin_entrenamiento']} / {R['cfg']['fin_validacion']}"
      f" | presupuesto {R['cfg']['entorno']['presupuesto_por_periodo']}")
# Evidencia de integración: cuántos municipios cruzan y con qué fuente entró cada reto.
print("cruzan catálogo/fuente oficial:", rep.get("cruzan"), "| solo catálogo:", len(rep.get("solo_catalogo", [])),
      "| solo oficial:", len(rep.get("solo_oficial", [])))
print("supuestos:", rep.get("supuestos", [])[:4])

def cb(ep, tot, fila, mejor, seg):
    if ep % 25 == 0 or ep == tot:
        print(f"  ep {ep}/{tot} · captura val {fila['captura_val']} · mejor {mejor:.3f} · {seg:.0f}s")

t0 = time.time()
agente, info = servicio.entrenar(R["D"], R["cfg"]["entorno"], {"episodios": EPISODIOS},
                                 callback=cb, huella=R["huella"])
SALIDA.write_bytes(servicio.exportar(agente, info))
print(f"agente_sismico.npz -> {SALIDA} | {SALIDA.stat().st_size:,} bytes"
      f" | mejor episodio: {info['episodio_mejor']} | captura val: {info['captura_val']:.3f}"
      f" | {time.time() - t0:.0f}s")

# Comparación con reglas simples en años no vistos (evidencia para el pitch).
ev = servicio.evaluar(agente, R["D"], R["cfg"]["entorno"])
print(ev["tabla"].round(3).to_string(index=False))
