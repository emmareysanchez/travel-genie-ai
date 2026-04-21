"""
travel_agent.py
---------------
Agente ReAct (Reasoning + Acting) para planificación de viajes.
Modelo base: Google Gemma 4 (importado con transformers)

Arquitectura ReAct:
  THOUGHT → ACTION → OBSERVATION → THOUGHT → ... → FINAL ANSWER

El agente razona paso a paso, decide qué tool usar, la ejecuta,
observa el resultado y repite hasta tener respuesta completa.
"""

import json
import re
import logging
import time
from typing import Any
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Importación lazy de tools (se registran dinámicamente) ---
from tools.flights import search_flights
from tools.hotels import search_hotels
from tools.transport import get_airport_to_hotel_transport
from tools.places_of_interest import search_places_of_interest

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. REGISTRO DE TOOLS
# ---------------------------------------------------------------------------
# Cada tool se describe con el esquema que el LLM necesita para invocarla.
# El agente parsea texto libre, así que este esquema sirve como documentación
# inyectada en el system prompt (no como function-calling nativo).

@dataclass
class Tool:
    name: str                    # identificador exacto que usará el LLM
    description: str             # qué hace la tool
    parameters: dict             # esquema de parámetros (tipo JSON Schema simplificado)
    callable: Any                # función Python real a ejecutar

TOOLS: list[Tool] = [
    Tool(
        name="search_flights",
        description=(
            "Busca vuelos disponibles entre un origen y un destino para una fecha dada. "
            "Si se proporciona return_date, busca viaje de ida y vuelta. "
            "Devuelve una lista de vuelos con aerolínea, horarios y precio estimado."
        ),
        parameters={
            "origin": "string — nombre de la ciudad de origen",
            "destination": "string — nombre de la ciudad de destino",
            "date": "string — fecha de salida en formato YYYY-MM-DD",
            "return_date": "string (opcional) — fecha de vuelta en formato YYYY-MM-DD",
            "passengers": "int (opcional, default 1) — número de pasajeros",
        },
        callable=search_flights,
    ),
    Tool(
        name="search_hotels",
        description=(
            "Busca hoteles disponibles en una ciudad para un rango de fechas. "
            "Devuelve opciones con nombre, rating, dirección geocodable y precio."
        ),
        parameters={
            "destination": "string — ciudad de destino (ej: Barcelona)",
            "check_in": "string — fecha de entrada YYYY-MM-DD",
            "check_out": "string — fecha de salida YYYY-MM-DD",
            "guests": "int (opcional, default 1) — número de huéspedes",
        },
        callable=search_hotels,
    ),
    
    # Tool(
    #     name="search_airport_transport",
    #     description=(
    #         "Calcula una ruta entre el aeropuerto de llegada y la dirección del hotel. "
    #         "Devuelve distancia y duración estimadas para un modo de transporte concreto. "
    #         "Si el usuario no especifica un tipo de transporte, llama esta función TRES VECES "
    #         "con los modos 'drive', 'bicycle' y 'transit' respectivamente, y presenta las tres opciones."
    #     ),
    #     parameters={
    #         "airport": "string — código IATA del aeropuerto (ej: BCN, MAD, JFK)",
    #         "hotel": "string — dirección completa del hotel (calle, número, ciudad, país)",
    #         "transport_type": "string (opcional, default 'drive') — modo de transporte: drive, bicycle, transit",
    #     },
    #     callable=get_airport_to_hotel_transport,
    # ),
    Tool(
        name="search_places_of_interest",
        description=(
            "Busca lugares de interés cercanos a una ubicación. "
            "Puede usarse con una ciudad, una dirección o coordenadas 'lat,lon'. "
            "Permite filtrar por categorías como museos, monumentos, restaurantes, "
            "parques, bares, vida_nocturna, transporte, etc."
        ),
        parameters={
            "location": "string — ciudad, dirección o coordenadas 'lat,lon'",
            "interest_types": "list[string] — tipos de interés, por ejemplo ['museos', 'monumentos'] o ['restaurantes', 'bares']",
            "radius_meters": "int (opcional, default 2000) — radio de búsqueda en metros",
            "limit": "int (opcional, default 5) — número máximo de resultados",
            "conditions": "list[string] (opcional) — filtros extra como ['wheelchair.yes'] o ['internet_access.free']",
            "lang": "string (opcional, default 'es') — idioma de la respuesta",
        },
        callable=search_places_of_interest,
    ),
]

# Índice rápido por nombre para el dispatch
TOOL_MAP: dict[str, Tool] = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 2. SYSTEM PROMPT
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    tools_docs = "\n\n".join(
        f"### Tool: `{t.name}`\n"
        f"**Descripción:** {t.description}\n"
        f"**Parámetros:**\n"
        + "\n".join(f"  - `{k}`: {v}" for k, v in t.parameters.items())
        for t in TOOLS
    )

    return f"""Eres un agente experto en planificación de viajes. Tu objetivo es ayudar al usuario a organizar su viaje de forma conversacional, clara y progresiva.

## Instrucciones de razonamiento (ReAct)

Debes razonar y actuar siguiendo ESTRICTAMENTE este formato en cada turno:

Thought: <tu razonamiento interno sobre qué necesitas hacer y por qué>
Action: <nombre_exacto_de_la_tool>
Action Input: <JSON válido con los parámetros de la tool>

Cuando recibas el resultado de una tool, éste aparecerá como:
Observation: <resultado de la tool>

Repite el ciclo Thought/Action/Action Input tantas veces como sea necesario.

Cuando tengas TODA la información necesaria para responder al usuario, usa:

Thought: Ya tengo toda la información. Voy a elaborar la respuesta final.
Final Answer: <respuesta completa, clara y bien formateada para el usuario>

## Reglas importantes

1. NUNCA inventes datos. Usa siempre las tools cuando necesites información externa.
2. Usa exactamente los nombres de tools indicados.
3. El JSON de Action Input debe ser válido. Usa comillas dobles.
4. Si faltan datos clave para hacer una búsqueda útil, NO llames todavía a tools. Haz primero una pregunta breve y útil al usuario.
5. Haz solo UNA pregunta por turno. No hagas listas largas de preguntas.
6. Intenta aclarar de forma progresiva estas preferencias solo si faltan o son relevantes:
   - presupuesto aproximado
   - si prefiere vuelos directos o más baratos
   - tipo de alojamiento o zona deseada
   - intereses principales del viaje
7. Si el usuario ya ha dado suficiente información, no preguntes más y pasa a usar las tools.
8. Si el usuario proporciona fecha de vuelta, úsala en search_flights como return_date.
9. No menciones transporte aeropuerto-hotel ni prometas opciones de transporte si esa tool no está disponible.
10. La respuesta final debe ser BREVE, priorizada y fácil de leer.
11. En la respuesta final incluye como máximo:
   - 1 vuelo recomendado y 1 alternativa
   - 1 hotel recomendado y 1 alternativa
   - hasta 3 lugares de interés
12. No vuelques listas enormes de resultados. Resume y selecciona.
13. Si el usuario pide más detalle, amplíalo en el siguiente turno.

## Cómo decidir cuándo preguntar

Pregunta antes de buscar si faltan datos esenciales como:
- origen y destino
- fecha de salida
- número de viajeros

También puedes preguntar antes de buscar si la consulta es demasiado abierta, por ejemplo:
- "Quiero un viaje a Italia"
- "Planea una escapada"
- "Busco vacaciones baratas"

En cambio, si el usuario ya da una petición suficientemente concreta, busca directamente.

## Estilo de respuesta final

La respuesta final debe seguir este estilo:

Final Answer: Aquí va mi recomendación para tu viaje:

Vuelo recomendado:
- aerolínea, horario principal, precio y escalas

Alternativa:
- aerolínea, horario principal, precio y escalas

Hotel recomendado:
- nombre, zona o dirección, precio por noche y valoración

Alternativa:
- nombre, zona o dirección, precio por noche y valoración

3 lugares que te encajan:
- lugar 1
- lugar 2
- lugar 3

Cierra con una frase breve ofreciendo continuar, por ejemplo:
"Si quieres, ahora te comparo solo los vuelos" o "Si quieres, te ajusto el plan a un presupuesto concreto".

## Tools disponibles

{tools_docs}

## Ejemplo 1: caso con información suficiente

Thought: El usuario quiere viajar de Madrid a Roma del 15 al 20 de junio para 2 personas y le interesan museos. Ya tengo suficiente información para empezar por vuelos.
Action: search_flights
Action Input: {{"origin": "Madrid", "destination": "Roma", "date": "2026-06-15", "return_date": "2026-06-20", "passengers": 2}}
Observation: [resultado de la búsqueda de vuelos]

Thought: Ahora busco hotel en Roma para esas fechas.
Action: search_hotels
Action Input: {{"destination": "Roma", "check_in": "2026-06-15", "check_out": "2026-06-20", "guests": 2}}
Observation: [resultado de hoteles]

Thought: Ahora busco lugares de interés relevantes en Roma.
Action: search_places_of_interest
Action Input: {{"location": "Roma, Italia", "interest_types": ["museos", "monumentos"], "radius_meters": 3000, "limit": 3, "lang": "es"}}
Observation: [resultado de lugares]

Thought: Ya tengo toda la información. Voy a elaborar una respuesta breve y priorizada.
Final Answer: Aquí va mi recomendación para tu viaje:
...

## Ejemplo 2: caso con información insuficiente

Thought: El usuario quiere una escapada a París, pero faltan fechas y número de viajeros. Primero haré una sola pregunta breve para concretar.
Final Answer: ¡Claro! Para proponerte opciones útiles, dime primero las fechas aproximadas y cuántas personas viajaríais.
"""


# ---------------------------------------------------------------------------
# 3. PARSER DE RESPUESTAS ReAct
# ---------------------------------------------------------------------------

@dataclass
class ReActStep:
    """Representa un paso parseado de la respuesta del LLM."""
    thought: str = ""
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None 

@dataclass
class TraceEvent:
    type: str
    content: str 
    
def _extract_balanced_json(text: str, label: str = "Action Input") -> str | None:
    marker = re.search(rf"{re.escape(label)}\s*:\s*", text, re.I)
    if not marker:
        return None

    start = text.find("{", marker.end())
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def parse_react_response(text: str) -> ReActStep:
    step = ReActStep()

    thought_match = re.search(
        r"Thought\s*:\s*(.+?)(?=Action\s*:|Final Answer\s*:|$)",
        text,
        re.S | re.I,
    )
    if thought_match:
        step.thought = thought_match.group(1).strip()

    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.S | re.I)
    if final_match:
        step.final_answer = final_match.group(1).strip()
        return step

    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.I)
    if action_match:
        step.action = action_match.group(1).strip()

    raw_json = _extract_balanced_json(text, "Action Input")
    if raw_json:
        try:
            step.action_input = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON inválido en Action Input: {e}\nRaw: {raw_json}")
            step.action_input = {}

    return step


# ---------------------------------------------------------------------------
# 4. EXECUTOR DE TOOLS
# ---------------------------------------------------------------------------
def _compact_result(result):
    if isinstance(result, list):
        result = result[:5]
        compact = []
        for item in result:
            if not isinstance(item, dict):
                compact.append(item)
                continue

            compact.append({
                k: item.get(k)
                for k in [
                    "airline",
                    "origin",
                    "destination",
                    "departure_date",
                    "return_date",
                    "departure_time",
                    "arrival_time",
                    "price",
                    "arrival_airport",
                    "duration",
                    "stops",
                    "name",
                    "price_per_night",
                    "rating",
                    "address",
                    "distance_meters",
                    "categories",
                    "duration_formatted",
                    "transport_type",
                ]
                if k in item
            })
        return compact

    if isinstance(result, dict):
        return {
            k: result.get(k)
            for k in [
                "airline",
                "origin",
                "destination",
                "departure_date",
                "return_date",
                "departure_time",
                "arrival_time",
                "price",
                "arrival_airport",
                "duration",
                "stops",
                "name",
                "price_per_night",
                "rating",
                "address",
                "distance_meters",
                "categories",
                "duration_formatted",
                "transport_type",
            ]
            if k in result
        }

    return result

def execute_tool(step: ReActStep) -> str:
    """
    Despacha la tool indicada en el step y devuelve la observación como string.
    Captura errores para que el agente pueda manejarlos.
    """
    tool_name = step.action
    tool = TOOL_MAP.get(tool_name)

    if tool is None:
        return f"[ERROR] Tool desconocida: '{tool_name}'. Tools disponibles: {list(TOOL_MAP.keys())}"

    params = step.action_input or {}
    logger.info(f"Ejecutando tool '{tool_name}' con params: {params}")

    try:
        result = tool.callable(**params)
        # Serializamos el resultado a JSON para que el LLM lo procese fácilmente
        if isinstance(result, (dict, list)):
            compact = _compact_result(result)
            return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        return str(result)
    except TypeError as e:
        return f"[ERROR] Parámetros incorrectos para '{tool_name}': {e}"
    except Exception as e:
        logger.error(f"Error ejecutando tool '{tool_name}': {e}")
        return f"[ERROR] Fallo al ejecutar '{tool_name}': {e}"


# ---------------------------------------------------------------------------
# 5. AGENTE ReAct PRINCIPAL
# ---------------------------------------------------------------------------

@dataclass
class TravelAgent:
    """
    Agente ReAct para planificación de viajes.

    Atributos configurables:
    - model_id: identificador del modelo HF
    - max_iterations: límite de ciclos Thought→Action→Observation
    - temperature: se mantiene por compatibilidad, aunque con do_sample=False no se usa
    - max_new_tokens: longitud máxima de generación
    """

    model_id: str = "google/gemma-4-E4B-it"
    # model_id: str =  "meta-llama/Llama-3.2-3B-Instruct" # "Qwen/Qwen2.5-1.5B-Instruct"  "Qwen/Qwen2.5-3B-Instruct" 
    max_iterations: int = 8 # vuelos(1) + hotel(1) + transporte×3(3) + razonamiento intermedio
    temperature: float = 0.2
    max_new_tokens: int = 700 # 400 # 1000 / 512

    _messages: list[dict] = field(default_factory=list, init=False)
    _tokenizer: Any = field(default=None, init=False)
    _model: Any = field(default=None, init=False)
    _device: str = field(default="cuda" if torch.cuda.is_available() else "cpu", init=False)

    def __post_init__(self):
        self._load_model()

        self._messages = [
            {"role": "system", "content": build_system_prompt()}
        ]

        logger.info(
            f"TravelAgent inicializado | modelo: {self.model_id} | "
            f"device: {self._device} | max_iter: {self.max_iterations}"
        )

    # ------------------------------------------------------------------
    # Carga del modelo
    # ------------------------------------------------------------------

    def _load_model(self):
        logger.info(f"Cargando modelo desde Hugging Face: {self.model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if not torch.cuda.is_available():
            self._model.to(self._device)

        self._model.eval()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> dict:
        logger.info(f"Usuario: {user_message}")
        self._messages.append({"role": "user", "content": user_message})

        trace: list[dict] = []

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteración ReAct {iteration}/{self.max_iterations} ---")
            trace.append({
                "type": "status",
                "content": f"Iteración {iteration}/{self.max_iterations}"
            })

            llm_response = self._call_llm()
            step = parse_react_response(llm_response)

            if step.thought:
                logger.info(f"Thought: {step.thought[:500]}...")
                trace.append({
                    "type": "thought",
                    "content": step.thought
                })

            if step.final_answer:
                logger.info("Final Answer recibida. Fin del bucle ReAct.")
                self._messages.append({"role": "assistant", "content": llm_response})

                trace.append({
                    "type": "final_answer",
                    "content": step.final_answer
                })

                return {
                    "final_answer": step.final_answer,
                    "trace": trace
                }

            if step.action:
                trace.append({
                    "type": "action",
                    "content": f"{step.action} | input={json.dumps(step.action_input or {}, ensure_ascii=False)}"
                })

                observation = execute_tool(step)
                logger.info(f"Observation ({step.action}): {observation[:200]}...")

                trace.append({
                    "type": "observation",
                    "content": observation[:1000]
                })

                self._messages.append({"role": "assistant", "content": llm_response})

                extra_note = ""
                if isinstance(observation, str) and observation.startswith("[ERROR]"):
                    extra_note = (
                        "\nSi el error es de parámetros, corrige Action Input. "
                        "Si falta información del usuario, pregúntala antes de seguir."
                    )

                self._messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}{extra_note}"
                })
            else:
                logger.warning("No se detectó Action ni Final Answer. Pidiendo al LLM que continúe.")
                trace.append({
                    "type": "warning",
                    "content": "El modelo no devolvió Action ni Final Answer."
                })

                self._messages.append({"role": "assistant", "content": llm_response})
                self._messages.append({
                    "role": "user",
                    "content": (
                        "Continúa con el siguiente paso del plan de viaje. "
                        "Recuerda usar el formato Thought/Action/Action Input o Final Answer."
                    )
                })

        fallback = (
            "Lo siento, no he podido completar la planificación del viaje en el número "
            "máximo de pasos. Por favor, intenta con una consulta más específica."
        )
        logger.error("max_iterations alcanzado sin Final Answer.")

        trace.append({
            "type": "error",
            "content": fallback
        })

        return {
            "final_answer": fallback,
            "trace": trace
        }

    def reset(self):
        """Reinicia el historial de conversación manteniendo el system prompt."""
        self._messages = [self._messages[0]]
        logger.info("Historial de conversación reiniciado.")

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _call_llm(self) -> str:
        """
        Genera la respuesta del modelo usando apply_chat_template + generate,
y devuelve el texto generado. Maneja errores de generación y decodificación.
        """
        try:
            text_input = self._tokenizer.apply_chat_template(
                self._messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_device = next(self._model.parameters()).device
            enc = self._tokenizer(text_input, return_tensors="pt").to(model_device)

            with torch.no_grad():
                logger.info(f"Mensajes en historial: {len(self._messages)}")
                logger.info(f"Tokens de entrada: {enc['input_ids'].shape[1]}")
                start = time.perf_counter()
                outputs = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
                elapsed = time.perf_counter() - start
                logger.info(f"Generación LLM completada en {elapsed:.2f} s")

            gen = outputs[0][enc["input_ids"].shape[1]:]
            response = self._tokenizer.decode(gen, skip_special_tokens=True).strip()

            return self._truncate_before_observation(response)

        except Exception as e:
            logger.error(f"Error en llamada al LLM: {e}")
            raise

    def _truncate_before_observation(self, text: str) -> str:
        """
        Si el LLM incluye un bloque de Observation en la misma generación, lo truncamos
        (no queremos que el LLM se invente una observación, sino que el agente la añada después de ejecutar la tool).
        """
        match = re.search(r"\bObservation\s*:", text, re.I)
        if match:
            return text[:match.start()].strip()
        return text


# ---------------------------------------------------------------------------
# 6. INTERFAZ DE LÍNEA DE COMANDOS (para pruebas rápidas)
# ---------------------------------------------------------------------------

def main():
    """CLI interactiva para testear el agente directamente."""
    print("=" * 60)
    print("  🌍 Travel Agent ReAct — powered by Gemma 4")
    print("  Escribe 'salir' para terminar, 'reset' para nueva conversación")
    print("=" * 60)

    agent = TravelAgent()

    while True:
        try:
            user_input = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta pronto!")
            break

        if not user_input:
            continue
        if user_input.lower() == "salir":
            print("¡Buen viaje! 👋")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[Conversación reiniciada]")
            continue

        response = agent.chat(user_input)
        print(f"\nAgente:\n{response}")


if __name__ == "__main__":
    main()