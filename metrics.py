"""
metrics.py
----------
Módulo de evaluación del agente de planificación de viajes.

Métricas implementadas
───────────────────────
Eficiencia (instrumentación del bucle ReAct):
  - latency_seconds      → tiempo total de la sesión
  - total_tokens         → tokens generados en todas las llamadas al LLM
  - tokens_per_second    → throughput medio
  - react_iterations     → ciclos Thought→Action usados
  - tool_calls           → llamadas totales a tools
  - tool_error_rate      → fracción de tool calls que devolvieron [ERROR]

Calidad objetiva (automática, sin coste):
  - completeness_score   → fracción de secciones esperadas presentes en la respuesta
  - date_coherence       → las fechas de la respuesta coinciden con las del request
  - geo_relevance        → el destino aparece en la respuesta
  - budget_respected     → el coste total no supera el presupuesto (si se dio)
  - flight_rank_score    → qué tan bueno es el vuelo elegido vs. candidatos (0-1)
  - hotel_rank_score     → qué tan bueno es el hotel elegido vs. candidatos (0-1)

Calidad semántica (LLM-as-judge, coste ~1 llamada por evaluación):
  - judge_relevance      → 1-5: ¿responde a lo que pidió el usuario?
  - judge_coherence      → 1-5: ¿son consistentes los datos entre sí?
  - judge_completeness   → 1-5: ¿incluye vuelo, hotel, transporte y POIs?
  - judge_reasoning      → frase de justificación del juez

Uso rápido
──────────
    from metrics import Evaluator, EvalResult

    evaluator = Evaluator(anthropic_api_key="sk-...")   # juez LLM opcional

    # 1. Instrumentar la sesión del agente
    with evaluator.session(trip_request) as session:
        final_answer = agent.chat(user_message)
        session.record_final_answer(final_answer)
        session.record_candidates(flights, hotels)
        session.record_selection(best_flight, best_hotel)

    result: EvalResult = session.evaluate()
    print(result.summary())
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATACLASS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    # ── Eficiencia ────────────────────────────────────────────────────────────
    latency_seconds: float = 0.0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    react_iterations: int = 0
    tool_calls: int = 0
    tool_error_rate: float = 0.0       # 0.0 – 1.0

    # ── Calidad objetiva ──────────────────────────────────────────────────────
    completeness_score: float = 0.0    # 0.0 – 1.0
    date_coherence: bool = False
    geo_relevance: bool = False
    budget_respected: bool = True      # True si no se proporcionó budget
    flight_rank_score: float = 0.0     # 0.0 – 1.0 (1 = mejor candidato elegido)
    hotel_rank_score: float = 0.0      # 0.0 – 1.0

    # ── Calidad semántica (LLM-as-judge) ──────────────────────────────────────
    judge_relevance: int = -1          # 1-5, -1 = no evaluado
    judge_coherence: int = -1
    judge_completeness: int = -1
    judge_reasoning: str = ""

    # ── Metadatos ─────────────────────────────────────────────────────────────
    request_summary: str = ""
    final_answer_length: int = 0

    # ── Scores agregados ─────────────────────────────────────────────────────

    @property
    def objective_quality_score(self) -> float:
        """
        Puntuación de calidad objetiva agregada (0.0 – 1.0).
        Promedio ponderado de las métricas objetivas.
        """
        components = [
            self.completeness_score,                          # peso 1
            float(self.date_coherence),                       # peso 1
            float(self.geo_relevance),                        # peso 1
            float(self.budget_respected),                     # peso 1
            self.flight_rank_score,                           # peso 1
            self.hotel_rank_score,                            # peso 1
        ]
        return sum(components) / len(components)

    @property
    def judge_score(self) -> float:
        """
        Puntuación del juez LLM normalizada a 0.0 – 1.0.
        Devuelve -1.0 si no se ejecutó el juez.
        """
        scores = [s for s in [self.judge_relevance, self.judge_coherence, self.judge_completeness] if s > 0]
        if not scores:
            return -1.0
        return (sum(scores) / len(scores) - 1) / 4  # normaliza [1-5] → [0-1]

    def summary(self) -> str:
        """Resumen legible de todos los resultados."""
        lines = [
            "═" * 60,
            "  EVALUACIÓN DEL AGENTE DE VIAJES",
            "═" * 60,
            f"  Petición : {self.request_summary}",
            "",
            "  ── Eficiencia ──────────────────────────────────────────",
            f"  Latencia total        : {self.latency_seconds:.2f} s",
            f"  Tokens generados      : {self.total_tokens}",
            f"  Throughput            : {self.tokens_per_second:.1f} tok/s",
            f"  Iteraciones ReAct     : {self.react_iterations}",
            f"  Tool calls            : {self.tool_calls}",
            f"  Tasa de errores tools : {self.tool_error_rate:.0%}",
            "",
            "  ── Calidad objetiva ────────────────────────────────────",
            f"  Completitud           : {self.completeness_score:.0%}",
            f"  Coherencia de fechas  : {'✓' if self.date_coherence else '✗'}",
            f"  Relevancia geográfica : {'✓' if self.geo_relevance else '✗'}",
            f"  Presupuesto respetado : {'✓' if self.budget_respected else '✗'}",
            f"  Rank vuelo elegido    : {self.flight_rank_score:.2f} / 1.00",
            f"  Rank hotel elegido    : {self.hotel_rank_score:.2f} / 1.00",
            f"  SCORE OBJETIVO        : {self.objective_quality_score:.2f} / 1.00",
        ]

        if self.judge_relevance > 0:
            lines += [
                "",
                "  ── LLM-as-judge ────────────────────────────────────────",
                f"  Relevancia            : {self.judge_relevance} / 5",
                f"  Coherencia            : {self.judge_coherence} / 5",
                f"  Completitud           : {self.judge_completeness} / 5",
                f"  SCORE JUEZ            : {self.judge_score:.2f} / 1.00",
                f"  Razonamiento          : {self.judge_reasoning}",
            ]

        lines.append("═" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialización completa a dict (útil para guardar resultados en JSON)."""
        d = asdict(self)
        d["objective_quality_score"] = self.objective_quality_score
        if self.judge_score >= 0:
            d["judge_score"] = self.judge_score
        return d


# ─────────────────────────────────────────────────────────────────────────────
# 2. MÉTRICAS OBJETIVAS (funciones puras, sin estado)
# ─────────────────────────────────────────────────────────────────────────────

# Secciones que debe contener una planificación completa
_EXPECTED_SECTIONS = ["vuelo", "hotel", "transporte", "lugar"]


def completeness_score(final_answer: str) -> float:
    """
    Fracción de secciones esperadas (vuelo, hotel, transporte, lugar)
    que están presentes en la respuesta final.

    >>> completeness_score("Vuelo Iberia · Hotel Central · transporte en metro · lugares de interés")
    1.0
    >>> completeness_score("Vuelo Iberia · Hotel Central")
    0.5
    """
    lower = final_answer.lower()
    hits = sum(1 for s in _EXPECTED_SECTIONS if s in lower)
    return hits / len(_EXPECTED_SECTIONS)


def date_coherence(final_answer: str, departure_date: str, return_date: str) -> bool:
    """
    Comprueba que las fechas del request (YYYY-MM-DD) aparecen en la respuesta.

    >>> date_coherence("Salida 2026-06-10, vuelta 2026-06-13", "2026-06-10", "2026-06-13")
    True
    >>> date_coherence("Salida 2026-07-01, vuelta 2026-07-05", "2026-06-10", "2026-06-13")
    False
    """
    found = set(re.findall(r"\d{4}-\d{2}-\d{2}", final_answer))
    return departure_date in found and return_date in found


def geo_relevance(final_answer: str, destination: str) -> bool:
    """
    Verifica que el destino solicitado aparece en la respuesta final.

    >>> geo_relevance("Tu hotel en París está en el centro", "París")
    True
    >>> geo_relevance("Tu hotel en Lyon está en el centro", "París")
    False
    """
    return destination.lower() in final_answer.lower()


def budget_respected(
    flight_price: float | None,
    hotel_total_price: float | None,
    budget: float | None,
) -> bool:
    """
    Si el usuario especificó un presupuesto, comprueba que la suma de vuelo
    + hotel no lo supera. Si no hay presupuesto, devuelve True.

    >>> budget_respected(150, 300, 500)
    True
    >>> budget_respected(150, 400, 500)
    False
    >>> budget_respected(150, 300, None)
    True
    """
    if budget is None:
        return True
    total = (flight_price or 0) + (hotel_total_price or 0)
    return total <= budget


def flight_rank_score(selected_flight: dict | None, all_flights: list[dict]) -> float:
    """
    Puntúa qué tan bueno es el vuelo seleccionado respecto al resto de candidatos.
    Criterio: precio más bajo = mejor.
    Score 1.0 si es el más barato, 0.0 si es el más caro.

    Devuelve 0.0 si no hay vuelo seleccionado o no hay candidatos.
    """
    if not selected_flight or not all_flights:
        return 0.0

    valid = [f for f in all_flights if isinstance(f.get("price"), (int, float))]
    if not valid:
        return 0.0

    sorted_asc = sorted(valid, key=lambda x: x["price"])

    # Buscamos el vuelo seleccionado por aerolínea + hora de salida
    def _key(f: dict) -> tuple:
        return (f.get("airline", ""), f.get("departure_time", ""), f.get("price"))

    selected_key = _key(selected_flight)
    try:
        rank = next(i for i, f in enumerate(sorted_asc) if _key(f) == selected_key)
    except StopIteration:
        # No encontrado exactamente; estimamos por precio
        sel_price = selected_flight.get("price", float("inf"))
        rank = sum(1 for f in sorted_asc if f["price"] < sel_price)

    n = len(sorted_asc)
    return 1.0 - rank / (n - 1) if n > 1 else 1.0


def hotel_rank_score(selected_hotel: dict | None, all_hotels: list[dict]) -> float:
    """
    Puntúa qué tan bueno es el hotel seleccionado respecto al resto de candidatos.
    Criterio: mayor ratio rating/precio = mejor (value for money).
    Score 1.0 si tiene mejor ratio, 0.0 si tiene peor ratio.

    Devuelve 0.0 si no hay hotel seleccionado o no hay candidatos.
    """
    if not selected_hotel or not all_hotels:
        return 0.0

    def _value(h: dict) -> float:
        price = h.get("price_per_night", 0)
        rating = h.get("rating", 0) or 0
        if price <= 0:
            return 0.0
        return rating / price

    valid = [h for h in all_hotels if isinstance(h.get("price_per_night"), (int, float)) and h["price_per_night"] > 0]
    if not valid:
        return 0.0

    sorted_desc = sorted(valid, key=_value, reverse=True)

    def _key(h: dict) -> str:
        return h.get("name", "") + str(h.get("price_per_night", ""))

    selected_key = _key(selected_hotel)
    try:
        rank = next(i for i, h in enumerate(sorted_desc) if _key(h) == selected_key)
    except StopIteration:
        sel_value = _value(selected_hotel)
        rank = sum(1 for h in sorted_desc if _value(h) > sel_value)

    n = len(sorted_desc)
    return 1.0 - rank / (n - 1) if n > 1 else 1.0


def tool_error_rate(observations: list[str]) -> float:
    """
    Fracción de observaciones de tools que contienen un error.

    >>> tool_error_rate(["[ERROR] algo", "ok", "[ERROR] otro"])
    0.6666...
    """
    if not observations:
        return 0.0
    errors = sum(1 for o in observations if o.strip().startswith("[ERROR]"))
    return errors / len(observations)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM-AS-JUDGE
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
Eres un evaluador experto en calidad de agentes de planificación de viajes.
Tu tarea es puntuar una respuesta generada por un agente IA dados:
  - la petición original del usuario
  - la respuesta final del agente

Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques
de código, sin backticks. Exactamente este esquema:
{
  "relevance": <entero 1-5>,
  "coherence": <entero 1-5>,
  "completeness": <entero 1-5>,
  "reasoning": "<una frase breve en español>"
}

Criterios de puntuación:
  relevance    → ¿Responde exactamente a lo que pidió el usuario (origen, destino, fechas, viajeros)?
  coherence    → ¿Son consistentes entre sí los datos (fechas, precios, destino, aeropuerto)?
  completeness → ¿Incluye vuelo, hotel, transporte y lugares de interés con datos concretos?

Escala:
  1 = muy deficiente  2 = deficiente  3 = aceptable  4 = bueno  5 = excelente
"""


def llm_judge(
    trip_request: dict,
    final_answer: str,
    anthropic_api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """
    Evalúa la respuesta final del agente usando un LLM como juez.

    Parámetros
    ----------
    trip_request      : dict con los campos de TripRequest (origin, destination, etc.)
    final_answer      : texto de la respuesta final del agente
    anthropic_api_key : clave API de Anthropic. Si es None, intenta leer ANTHROPIC_API_KEY del entorno.
    model             : modelo a usar como juez (por defecto claude-sonnet-4)

    Devuelve
    --------
    dict con claves: relevance (int), coherence (int), completeness (int), reasoning (str)
    Si falla, devuelve valores centinela (-1) y un mensaje de error en reasoning.
    """
    import os

    api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY no disponible. Saltando LLM-as-judge.")
        return {"relevance": -1, "coherence": -1, "completeness": -1, "reasoning": "Juez no disponible (sin API key)"}

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("Librería 'anthropic' no instalada. Saltando LLM-as-judge.")
        return {"relevance": -1, "coherence": -1, "completeness": -1, "reasoning": "Juez no disponible (anthropic no instalado)"}

    user_content = (
        f"PETICIÓN DEL USUARIO:\n{json.dumps(trip_request, ensure_ascii=False, indent=2)}\n\n"
        f"RESPUESTA DEL AGENTE:\n{final_answer}"
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=256,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = message.content[0].text.strip()
        result = json.loads(raw)
        # Validar estructura mínima
        for key in ("relevance", "coherence", "completeness", "reasoning"):
            if key not in result:
                raise ValueError(f"Clave faltante en respuesta del juez: {key}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"El juez no devolvió JSON válido: {e}\nRaw: {raw!r}")
        return {"relevance": -1, "coherence": -1, "completeness": -1, "reasoning": f"Error de parseo: {e}"}
    except Exception as e:
        logger.error(f"Error al llamar al juez LLM: {e}")
        return {"relevance": -1, "coherence": -1, "completeness": -1, "reasoning": f"Error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION CONTEXT MANAGER (instrumentación del agente)
# ─────────────────────────────────────────────────────────────────────────────

class EvalSession:
    """
    Contexto de evaluación para una sesión del agente.

    Registra eventos durante la ejecución del agente (tokens, iteraciones,
    tool calls, etc.) y calcula todas las métricas al final.

    Uso:
        session = EvalSession(trip_request, anthropic_api_key="sk-...")
        session.start()
        # ... ejecutar agente ...
        session.record_tokens(tokens_generated)
        session.record_tool_call(observation)
        session.record_iteration()
        session.record_final_answer(final_answer)
        session.record_candidates(flights, hotels)
        session.record_selection(best_flight, best_hotel)
        result = session.evaluate()
    """

    def __init__(self, trip_request: dict, anthropic_api_key: str | None = None, run_judge: bool = True):
        self.trip_request = trip_request
        self.anthropic_api_key = anthropic_api_key
        self.run_judge = run_judge

        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._total_tokens: int = 0
        self._iterations: int = 0
        self._observations: list[str] = []
        self._final_answer: str = ""
        self._all_flights: list[dict] = []
        self._all_hotels: list[dict] = []
        self._selected_flight: dict | None = None
        self._selected_hotel: dict | None = None

    @property
    def tool_calls_count(self) -> int:
        return len(self._observations)  # ajusta al nombre real del atributo interno

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        self._end_time = time.perf_counter()

    def record_tokens(self, n: int):
        """Llama una vez por cada respuesta del LLM con el número de tokens generados."""
        self._total_tokens += n

    def record_iteration(self):
        """Llama al inicio de cada iteración del bucle ReAct."""
        self._iterations += 1

    def record_tool_call(self, observation: str):
        """Llama después de cada ejecución de tool con la observación resultante."""
        self._observations.append(observation)

    def record_final_answer(self, text: str):
        """Llama con la respuesta final devuelta al usuario."""
        self._final_answer = text
        if self._end_time == 0.0:
            self.stop()

    def record_candidates(self, flights: list[dict], hotels: list[dict]):
        """
        Registra todos los candidatos devueltos por search_flights y search_hotels
        para calcular los rank scores.
        """
        self._all_flights = flights or []
        self._all_hotels = hotels or []

    def record_selection(self, flight: dict | None, hotel: dict | None):
        """Registra el vuelo y hotel que el agente eligió recomendar."""
        self._selected_flight = flight
        self._selected_hotel = hotel

    def evaluate(self) -> EvalResult:
        """Calcula y devuelve todas las métricas."""
        if self._end_time == 0.0:
            self.stop()

        latency = self._end_time - self._start_time
        tps = self._total_tokens / latency if latency > 0 else 0.0

        req = self.trip_request
        nights = _nights(req.get("departure_date", ""), req.get("return_date", ""))
        hotel_total = (
            (self._selected_hotel.get("price_per_night", 0) * nights)
            if self._selected_hotel else None
        )

        result = EvalResult(
            # Eficiencia
            latency_seconds=round(latency, 3),
            total_tokens=self._total_tokens,
            tokens_per_second=round(tps, 2),
            react_iterations=self._iterations,
            tool_calls=len(self._observations),
            tool_error_rate=round(tool_error_rate(self._observations), 4),
            # Calidad objetiva
            completeness_score=round(completeness_score(self._final_answer), 4),
            date_coherence=date_coherence(
                self._final_answer,
                req.get("departure_date", ""),
                req.get("return_date", ""),
            ),
            geo_relevance=geo_relevance(self._final_answer, req.get("destination", "")),
            budget_respected=budget_respected(
                self._selected_flight.get("price") if self._selected_flight else None,
                hotel_total,
                req.get("budget"),
            ),
            flight_rank_score=round(flight_rank_score(self._selected_flight, self._all_flights), 4),
            hotel_rank_score=round(hotel_rank_score(self._selected_hotel, self._all_hotels), 4),
            # Metadatos
            request_summary=(
                f"{req.get('origin','?')} → {req.get('destination','?')} "
                f"({req.get('departure_date','?')} / {req.get('return_date','?')}, "
                f"{req.get('travelers', 1)} viajero/s)"
            ),
            final_answer_length=len(self._final_answer),
        )

        # LLM-as-judge (opcional)
        if self.run_judge and self._final_answer:
            judge_result = llm_judge(
                trip_request=req,
                final_answer=self._final_answer,
                anthropic_api_key=self.anthropic_api_key,
            )
            result.judge_relevance = judge_result.get("relevance", -1)
            result.judge_coherence = judge_result.get("coherence", -1)
            result.judge_completeness = judge_result.get("completeness", -1)
            result.judge_reasoning = judge_result.get("reasoning", "")

        return result


class Evaluator:
    """
    Factoría de EvalSessions. Mantiene el historial de todas las evaluaciones
    de una misma instancia del agente para comparativas y exportación.

    Uso:
        evaluator = Evaluator(anthropic_api_key="sk-...", run_judge=True)
        with evaluator.session(trip_request) as session:
            answer = agent.chat(...)
            session.record_final_answer(answer)
        result = session.result   # EvalResult disponible fuera del with
    """

    def __init__(self, anthropic_api_key: str | None = None, run_judge: bool = True):
        self.anthropic_api_key = anthropic_api_key
        self.run_judge = run_judge
        self.history: list[EvalResult] = []

    @contextmanager
    def session(self, trip_request: dict):
        """Context manager que crea, inicia y evalúa automáticamente una sesión."""
        sess = EvalSession(
            trip_request=trip_request,
            anthropic_api_key=self.anthropic_api_key,
            run_judge=self.run_judge,
        )
        sess.start()
        try:
            yield sess
        finally:
            result = sess.evaluate()
            sess.result = result
            self.history.append(result)
            logger.info(f"Sesión evaluada:\n{result.summary()}")

    def export_history(self, path: str):
        """Exporta el historial completo de evaluaciones a un fichero JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.history], f, ensure_ascii=False, indent=2)
        logger.info(f"Historial exportado a {path}")

    def aggregate(self) -> dict:
        """
        Calcula estadísticas agregadas sobre todas las sesiones en el historial.
        Útil para comparar distintas configuraciones del agente.
        """
        if not self.history:
            return {}

        def _avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in self.history]
            return round(sum(vals) / len(vals), 4)

        return {
            "n_sessions": len(self.history),
            "avg_latency_seconds": _avg("latency_seconds"),
            "avg_total_tokens": _avg("total_tokens"),
            "avg_tokens_per_second": _avg("tokens_per_second"),
            "avg_react_iterations": _avg("react_iterations"),
            "avg_tool_error_rate": _avg("tool_error_rate"),
            "avg_completeness_score": _avg("completeness_score"),
            "pct_date_coherence": round(sum(r.date_coherence for r in self.history) / len(self.history), 4),
            "pct_geo_relevance": round(sum(r.geo_relevance for r in self.history) / len(self.history), 4),
            "pct_budget_respected": round(sum(r.budget_respected for r in self.history) / len(self.history), 4),
            "avg_flight_rank_score": _avg("flight_rank_score"),
            "avg_hotel_rank_score": _avg("hotel_rank_score"),
            "avg_objective_quality": round(
                sum(r.objective_quality_score for r in self.history) / len(self.history), 4
            ),
            "avg_judge_relevance": _avg("judge_relevance"),
            "avg_judge_coherence": _avg("judge_coherence"),
            "avg_judge_completeness": _avg("judge_completeness"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def _nights(departure: str, return_date: str) -> int:
    """Calcula el número de noches entre dos fechas YYYY-MM-DD."""
    try:
        from datetime import date
        fmt = "%Y-%m-%d"
        d1 = date.fromisoformat(departure)
        d2 = date.fromisoformat(return_date)
        return max((d2 - d1).days, 1)
    except (ValueError, TypeError):
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. CÓMO INTEGRAR EN travel_agent.py
# ─────────────────────────────────────────────────────────────────────────────
#
# En travel_agent.py, modifica el método chat() así:
#
#   def chat(self, user_message: str, eval_session: EvalSession | None = None) -> str:
#       ...
#       for iteration in range(1, self.max_iterations + 1):
#           if eval_session:
#               eval_session.record_iteration()
#
#           llm_response = self._call_llm()
#
#           # Contar tokens generados
#           if eval_session:
#               tokens = len(self._tokenizer.encode(llm_response))
#               eval_session.record_tokens(tokens)
#
#           step = parse_react_response(llm_response)
#
#           if step.final_answer:
#               if eval_session:
#                   eval_session.record_final_answer(step.final_answer)
#               return step.final_answer
#
#           if step.action:
#               observation = execute_tool(step)
#               if eval_session:
#                   eval_session.record_tool_call(observation)
#               ...
#
# Y en app.py (o en un script de test):
#
#   from metrics import Evaluator
#
#   evaluator = Evaluator(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
#
#   @app.post("/plan_trip")
#   def plan_trip(request: TripRequest):
#       req_dict = request.model_dump()
#       with evaluator.session(req_dict) as session:
#           answer = agent.chat(
#               f"Planifica un viaje de {request.origin} a {request.destination}...",
#               eval_session=session,
#           )
#           # Si tienes acceso a candidatos y selección:
#           session.record_candidates(last_flights, last_hotels)
#           session.record_selection(best_flight, best_hotel)
#       return {"answer": answer, "eval": session.result.to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# 7. EJEMPLO / SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)

    # ── Ejemplo con datos mock ────────────────────────────────────────────────
    from mocks import MOCK_FLIGHTS, MOCK_HOTELS

    mock_request = {
        "origin": "Madrid",
        "destination": "Paris",
        "departure_date": "2026-06-10",
        "return_date": "2026-06-13",
        "travelers": 1,
        "budget": 600.0,
    }

    mock_answer = (
        "¡Aquí tienes tu plan de viaje de Madrid a París!\n\n"
        "✈️ Vuelo: Iberia, salida 08:30, llegada 10:25, precio 145 €\n"
        "Fechas: 2026-06-10 → 2026-06-13\n\n"
        "🏨 Hotel: Hotel Central Paris, 120 €/noche, rating 8.7\n"
        "Dirección: 12 Rue du Centre, Paris\n\n"
        "🚌 Transporte: RER B desde CDG, ~45 min, ~12 €\n\n"
        "📍 Lugares de interés:\n"
        "  • Torre Eiffel — monumento\n"
        "  • Museo del Louvre — museo\n"
        "  • Montmartre — barrio histórico\n"
    )

    session = EvalSession(mock_request, run_judge=False)
    session.start()

    # Simulamos 3 iteraciones y 4 tool calls (1 error)
    session.record_iteration()
    session.record_tokens(320)
    session.record_tool_call('[{"airline": "Iberia", "price": 145}]')

    session.record_iteration()
    session.record_tokens(280)
    session.record_tool_call('[{"name": "Hotel Central Paris", "price_per_night": 120, "rating": 8.7}]')

    session.record_iteration()
    session.record_tokens(190)
    session.record_tool_call("[ERROR] API timeout")

    session.record_final_answer(mock_answer)
    session.record_candidates(MOCK_FLIGHTS, MOCK_HOTELS)
    session.record_selection(MOCK_FLIGHTS[0], MOCK_HOTELS[0])

    result = session.evaluate()
    print(result.summary())
    print("\nDict completo:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
