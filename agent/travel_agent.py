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
from typing import Any
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Importación lazy de tools (se registran dinámicamente) ---
from tools.flights import search_flights
from tools.hotels import search_hotels
from tools.transport import get_airport_to_hotel_transport

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
    Tool(
        name="search_airport_transport",
        description=(
            "Calcula una ruta entre el aeropuerto de llegada y la dirección del hotel. "
            "Devuelve distancia y duración estimadas para un modo de transporte concreto."
        ),
        parameters={
            "airport": "string — código IATA del aeropuerto (ej: BCN)",
            "hotel": "string — dirección completa del hotel o dirección geocodable",
            "transport_type": "string (opcional, default 'drive') — modo de transporte: drive, transit, walk, bicycle",
        },
        callable=get_airport_to_hotel_transport,
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

    return f"""Eres un agente experto en planificación de viajes. Tu objetivo es ayudar al usuario a organizar su viaje de forma completa: vuelos, alojamiento y transporte desde el aeropuerto.

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

1. NUNCA inventes datos de vuelos, hoteles o transporte. Usa siempre las tools.
2. Usa exactamente los nombres de tools indicados.
3. El JSON de Action Input debe ser válido. Usa comillas dobles.
4. Si el usuario no proporciona algún dato necesario, pregúntale antes de llamar a la tool.
5. Busca primero vuelos, luego hotel y después transporte.
6. Si el usuario proporciona fecha de vuelta, úsala en search_flights como return_date.
7. Para search_airport_transport, usa la dirección geocodable del hotel en el campo hotel.
8. Cuando ya tengas toda la información, devuelve una respuesta final clara con la mejor combinación encontrada.

## Tools disponibles

{tools_docs}

## Ejemplo de flujo

Thought: El usuario quiere viajar de Madrid a Roma del 15 al 20 de junio. Primero buscaré vuelos.
Action: search_flights
Action Input: {{"origin": "Madrid", "destination": "Roma", "date": "2026-06-15", "return_date": "2026-06-20", "passengers": 1}}
Observation: [resultado de la búsqueda de vuelos]

Thought: Ahora busco hotel en Roma para esas fechas.
Action: search_hotels
Action Input: {{"destination": "Roma", "check_in": "2026-06-15", "check_out": "2026-06-20", "guests": 1}}
Observation: [resultado de hoteles]

Thought: Ahora calculo la ruta desde el aeropuerto al hotel.
Action: search_airport_transport
Action Input: {{"airport": "FCO", "hotel": "Hotel Colosseo, Roma, Italia", "transport_type": "drive"}}
Observation: [resultado de transporte]

Thought: Ya tengo toda la información. Voy a elaborar la respuesta final.
Final Answer: Aquí tienes tu plan de viaje completo...
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
            return json.dumps(result, ensure_ascii=False, indent=2)
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
    max_iterations: int = 6 # 10
    temperature: float = 0.2
    max_new_tokens: int = 400 # 1000 / 512

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

    def chat(self, user_message: str) -> str:
        """
        Punto de entrada principal. Recibe el mensaje del usuario y ejecuta
        el bucle ReAct hasta obtener una Final Answer o alcanzar max_iterations.

        Returns:
            La respuesta final para mostrar al usuario.
        """
        logger.info(f"Usuario: {user_message}")
        self._messages.append({"role": "user", "content": user_message})

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteración ReAct {iteration}/{self.max_iterations} ---")

            # 1. Llamada al LLM
            llm_response = self._call_llm()
            logger.debug(f"LLM raw response:\n{llm_response}")

            # 2. Parsear la respuesta
            step = parse_react_response(llm_response)
            logger.info(f"Thought: {step.thought[:2500]}...")

            # 3. Si hay Final Answer, terminamos
            if step.final_answer:
                logger.info("Final Answer recibida. Fin del bucle ReAct.")
                # Añadimos la respuesta al historial para conversaciones multi-turno
                self._messages.append({"role": "assistant", "content": llm_response})
                return step.final_answer

            # 4. Si hay una Action, ejecutamos la tool
            if step.action:
                observation = execute_tool(step)
                logger.info(f"Observation ({step.action}): {observation[:200]}...")

                # Añadimos el turno del asistente y la observación al historial
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
                # El LLM no siguió el formato esperado
                logger.warning("No se detectó Action ni Final Answer. Pidiendo al LLM que continúe.")
                self._messages.append({"role": "assistant", "content": llm_response})
                self._messages.append({
                    "role": "user",
                    "content": (
                        "Continúa con el siguiente paso del plan de viaje. "
                        "Recuerda usar el formato Thought/Action/Action Input o Final Answer."
                    )
                })

        # Límite de iteraciones alcanzado
        fallback = (
            "Lo siento, no he podido completar la planificación del viaje en el número "
            "máximo de pasos. Por favor, intenta con una consulta más específica."
        )
        logger.error("max_iterations alcanzado sin Final Answer.")
        return fallback

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
                outputs = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

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