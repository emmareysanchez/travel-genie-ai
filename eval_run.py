"""
eval_run.py
-----------
Script de evaluación offline del agente de planificación de viajes.

Uso
───
    python eval_run.py                        # evaluación completa con juez LLM
    python eval_run.py --no-judge             # sin LLM-as-judge (más rápido, sin coste)
    python eval_run.py --output resultados.json

Qué hace
────────
1. Define una batería de casos de prueba fijos y reproducibles.
2. Lanza el agente sobre cada caso exactamente como lo haría un usuario real.
3. Calcula todas las métricas (eficiencia + calidad objetiva + LLM-as-judge).
4. Imprime un resumen por consola y guarda los resultados en JSON.

Por qué es útil
───────────────
Cada vez que cambies algo en el agente (modelo, system prompt, max_new_tokens,
lógica de selección de vuelos...) vuelves a ejecutar este script y comparas
los números. Así sabes con evidencia cuantitativa si el cambio mejora o empeora
el comportamiento, sin tener que revisar respuestas manualmente una por una.
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from agent.travel_agent_metrics import TravelAgent
from metrics import Evaluator, EvalResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # silenciamos logs del agente durante la eval
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BATERÍA DE CASOS DE PRUEBA
# ---------------------------------------------------------------------------
# Cada caso representa una petición de usuario distinta.
# Son fijos e invariables: si los cambias entre ejecuciones, no puedes comparar.
#
# Cubren distintos escenarios:
#   - viaje corto vs. largo
#   - con y sin presupuesto
#   - 1 viajero vs. varios
#   - destinos europeos e intercontinentales

TEST_CASES = [
    # ── Caso 1: viaje estándar europeo, sin presupuesto ──────────────────────
    {
        "id": "TC01",
        "description": "Viaje corto a París, 1 viajero, sin presupuesto",
        "request": {
            "origin": "Madrid",
            "destination": "París",
            "departure_date": "2026-09-10",
            "return_date": "2026-09-13",
            "travelers": 1,
            "budget": None,
        },
    },
    # ── Caso 2: viaje con presupuesto ajustado ────────────────────────────────
    {
        "id": "TC02",
        "description": "Viaje a Roma con presupuesto de 500 €",
        "request": {
            "origin": "Madrid",
            "destination": "Roma",
            "departure_date": "2026-10-05",
            "return_date": "2026-10-10",
            "travelers": 1,
            "budget": 500.0,
        },
    },
    # ── Caso 3: viaje en grupo ────────────────────────────────────────────────
    {
        "id": "TC03",
        "description": "Viaje a Lisboa para 2 viajeros",
        "request": {
            "origin": "Madrid",
            "destination": "Lisboa",
            "departure_date": "2026-11-20",
            "return_date": "2026-11-24",
            "travelers": 2,
            "budget": None,
        },
    },
    # ── Caso 4: destino de larga distancia ───────────────────────────────────
    {
        "id": "TC04",
        "description": "Viaje intercontinental a Nueva York",
        "request": {
            "origin": "Madrid",
            "destination": "Nueva York",
            "departure_date": "2026-12-20",
            "return_date": "2026-12-30",
            "travelers": 1,
            "budget": 2000.0,
        },
    },
    # ── Caso 5: estancia larga ────────────────────────────────────────────────
    {
        "id": "TC05",
        "description": "Estancia larga en Berlín, presupuesto medio",
        "request": {
            "origin": "Barcelona",
            "destination": "Berlín",
            "departure_date": "2026-08-01",
            "return_date": "2026-08-15",
            "travelers": 1,
            "budget": 1200.0,
        },
    },
]


# ---------------------------------------------------------------------------
# CONSTRUCCIÓN DEL MENSAJE DE USUARIO
# ---------------------------------------------------------------------------

def build_user_message(req: dict) -> str:
    """Genera el mensaje de usuario a partir del request, igual que haría app.py."""
    msg = (
        f"Planifica un viaje de {req['origin']} a {req['destination']} "
        f"del {req['departure_date']} al {req['return_date']} "
        f"para {req.get('travelers', 1)} viajero/s."
    )
    if req.get("budget"):
        msg += f" Presupuesto máximo: {req['budget']} €."
    return msg

# ---------------------------------------------------------------------------
# PERMITIR INTERACCIÓN CON EL USUARIO EVALUADOR
# ---------------------------------------------------------------------------

def run_human_conversation(agent, test_case: dict, eval_session, max_turns: int = 5) -> str:
    """
    Ejecuta una conversación completa con el agente en modo humano.
    El evaluador responde manualmente cada vez que el agente pregunta.
    Termina cuando el agente emite una Final Answer con tool calls reales.
    """
    # initial_message = test_case["input"]
    initial_message = test_case
    print(f"\n[USUARIO → AGENTE] {initial_message}")

    result = agent.chat(initial_message, eval_session=eval_session)

    if _has_real_tool_calls(eval_session):
        return result

    for turn in range(max_turns):
        print(f"\n[AGENTE → USUARIO] {result}")
        user_response = input("\n  Tu respuesta: ").strip()
        result = agent.chat(user_response, eval_session=eval_session)

        if _has_real_tool_calls(eval_session):
            break

    return result


def _has_real_tool_calls(eval_session) -> bool:
    """Comprueba si la sesión ya registró al menos una tool call real."""
    # Ajusta según cómo EvalSession exponga sus datos internos
    return getattr(eval_session, "tool_calls_count", 0) > 0


# ---------------------------------------------------------------------------
# RUNNER PRINCIPAL
# ---------------------------------------------------------------------------

def run_evaluation(
    output_path: str = "resultados_eval.json",
    run_judge: bool = True,
) -> list[EvalResult]:
    """
    Ejecuta la batería de tests completa y devuelve la lista de EvalResult.

    Parámetros
    ----------
    output_path : ruta del fichero JSON donde guardar los resultados
    run_judge   : si True, llama al LLM-as-judge para cada caso (requiere ANTHROPIC_API_KEY)
    """

    print("=" * 65)
    print("  EVALUACIÓN DEL AGENTE DE PLANIFICACIÓN DE VIAJES")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Casos de prueba : {len(TEST_CASES)}")
    print(f"  LLM-as-judge    : {'activado' if run_judge else 'desactivado'}")
    print("=" * 65)

    # Cargar el agente una sola vez (la carga del modelo es costosa)
    print("\n⏳ Cargando modelo...", flush=True)
    agent = TravelAgent()
    print("✓ Modelo cargado.\n")

    evaluator = Evaluator(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        run_judge=run_judge,
    )

    results: list[EvalResult] = []

    for i, case in enumerate(TEST_CASES, 1):
        case_id = case["id"]
        description = case["description"]
        req = case["request"]

        print(f"[{i}/{len(TEST_CASES)}] {case_id}: {description}")
        print(f"         {req['origin']} → {req['destination']} "
              f"({req['departure_date']} / {req['return_date']})", flush=True)

        user_message = build_user_message(req)

        try:
            with evaluator.session(req) as session:
                # agent.chat(user_message, eval_session=session)
                run_human_conversation(agent, user_message, eval_session=session)

            result: EvalResult = session.result
            results.append(result)

            # Resumen compacto por línea
            judge_str = (
                f"  juez={result.judge_score:.2f}"
                if result.judge_score >= 0
                else "  juez=n/a"
            )
            print(
                f"         ✓  latencia={result.latency_seconds:.1f}s  "
                f"tokens={result.total_tokens}  "
                f"iter={result.react_iterations}  "
                f"calidad={result.objective_quality_score:.2f}"
                f"{judge_str}"
            )

        except Exception as e:
            logger.error(f"Error en caso {case_id}: {e}")
            print(f"         ✗  ERROR: {e}")

        # Resetear historial entre casos para que no se contaminen
        agent.reset()
        print()

    # ── Resumen agregado ──────────────────────────────────────────────────────
    if results:
        print("=" * 65)
        print("  RESUMEN AGREGADO")
        print("=" * 65)
        agg = evaluator.aggregate()
        _print_aggregate(agg)

    # ── Guardar a disco ───────────────────────────────────────────────────────
    evaluator.export_history(output_path)
    print(f"\n💾 Resultados guardados en: {Path(output_path).resolve()}")

    return results


# ---------------------------------------------------------------------------
# PRETTY PRINT DEL AGREGADO
# ---------------------------------------------------------------------------

def _print_aggregate(agg: dict):
    if not agg:
        print("  (sin datos)")
        return

    rows = [
        ("Sesiones evaluadas",      f"{agg['n_sessions']}"),
        ("── Eficiencia",           ""),
        ("  Latencia media",        f"{agg['avg_latency_seconds']:.2f} s"),
        ("  Tokens medios",         f"{agg['avg_total_tokens']:.0f}"),
        ("  Throughput medio",      f"{agg['avg_tokens_per_second']:.1f} tok/s"),
        ("  Iteraciones ReAct",     f"{agg['avg_react_iterations']:.1f}"),
        ("  Tasa errores tools",    f"{agg['avg_tool_error_rate']:.0%}"),
        ("── Calidad objetiva",     ""),
        ("  Completitud",           f"{agg['avg_completeness_score']:.0%}"),
        ("  Coherencia fechas",     f"{agg['pct_date_coherence']:.0%}"),
        ("  Relevancia geográfica", f"{agg['pct_geo_relevance']:.0%}"),
        ("  Presupuesto respetado", f"{agg['pct_budget_respected']:.0%}"),
        ("  Rank vuelo",            f"{agg['avg_flight_rank_score']:.2f} / 1.00"),
        ("  Rank hotel",            f"{agg['avg_hotel_rank_score']:.2f} / 1.00"),
        ("  SCORE OBJETIVO",        f"{agg['avg_objective_quality']:.2f} / 1.00"),
    ]

    if agg.get("avg_judge_relevance", -1) >= 0:
        rows += [
            ("── LLM-as-judge",     ""),
            ("  Relevancia",        f"{agg['avg_judge_relevance']:.2f} / 5"),
            ("  Coherencia",        f"{agg['avg_judge_coherence']:.2f} / 5"),
            ("  Completitud",       f"{agg['avg_judge_completeness']:.2f} / 5"),
        ]

    for label, value in rows:
        if value == "":
            print(f"\n  {label}")
        else:
            print(f"  {label:<28} {value}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa el agente de viajes sobre una batería de casos de prueba."
    )
    parser.add_argument(
        "--output", "-o",
        default="resultados_eval.json",
        help="Ruta del fichero JSON de salida (default: resultados_eval.json)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Desactiva el LLM-as-judge (más rápido, sin coste de API)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_evaluation(
        output_path=args.output,
        run_judge=not args.no_judge,
    )
    sys.exit(0 if results else 1)
