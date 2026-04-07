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
            "Devuelve una lista de vuelos con aerolínea, horarios y precio estimado."
        ),
        parameters={
            "origin": "string — código IATA del aeropuerto de origen (ej: MAD)",
            "destination": "string — código IATA del aeropuerto de destino (ej: BCN)",
            "date": "string — fecha de salida en formato YYYY-MM-DD",
            "passengers": "int (opcional, default 1) — número de pasajeros",
        },
        callable=search_flights,
    ),
    Tool(
        name="search_hotels",
        description=(
            "Busca hoteles disponibles en una ciudad para un rango de fechas. "
            "Devuelve opciones con nombre, categoría, ubicación y precio por noche."
        ),
        parameters={
            "city": "string — ciudad de destino (ej: Barcelona)",
            "check_in": "string — fecha de entrada YYYY-MM-DD",
            "check_out": "string — fecha de salida YYYY-MM-DD",
            "guests": "int (opcional, default 1) — número de huéspedes",
        },
        callable=search_hotels,
    ),
    Tool(
        name="search_airport_transport",
        description=(
            "Busca opciones de transporte entre el aeropuerto de llegada y un hotel "
            "o zona de la ciudad. Devuelve taxi, shuttle, metro u otras alternativas "
            "con duración y coste estimado."
        ),
        parameters={
            "airport": "string — código IATA del aeropuerto (ej: BCN)",
            "destination": "string — nombre del hotel o zona de destino",
            "datetime": "string — fecha y hora de llegada YYYY-MM-DD HH:MM",
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
    """
    Construye el system prompt que instruye al LLM a seguir el patrón ReAct.
    Incluye la documentación de todas las tools disponibles.
    """
    tools_docs = "\n\n".join(
        f"### Tool: `{t.name}`\n"
        f"**Descripción:** {t.description}\n"
        f"**Parámetros:**\n"
        + "\n".join(f"  - `{k}`: {v}" for k, v in t.parameters.items())
        for t in TOOLS
    )

    return f"""Eres un agente experto en planificación de viajes. Tu objetivo es ayudar al usuario \
a organizar su viaje de forma completa: vuelos, alojamiento y transporte desde el aeropuerto.

## Instrucciones de razonamiento (ReAct)

Debes razonar y actuar siguiendo ESTRICTAMENTE este formato en cada turno:

```
Thought: <tu razonamiento interno sobre qué necesitas hacer y por qué>
Action: <nombre_exacto_de_la_tool>
Action Input: <JSON válido con los parámetros de la tool>
```

Cuando recibas el resultado de una tool, éste aparecerá como:
```
Observation: <resultado de la tool>
```

Repite el ciclo Thought/Action/Action Input tantas veces como sea necesario.
Cuando tengas TODA la información necesaria para responder al usuario, usa:

```
Thought: Ya tengo toda la información. Voy a elaborar la respuesta final.
Final Answer: <respuesta completa, clara y bien formateada para el usuario>
```

## Reglas importantes

1. NUNCA inventes datos de vuelos, hoteles o transporte. Usa siempre las tools.
2. Usa exactamente los nombres de tools indicados (search_flights, search_hotels, search_airport_transport).
3. El JSON de Action Input debe ser válido. Usa comillas dobles.
4. Si el usuario no proporciona algún dato necesario, pregúntale antes de llamar a la tool.
5. Coordina las búsquedas de forma lógica: primero vuelos, luego hotel, luego transporte.
6. En la Final Answer, presenta un resumen ejecutivo con las mejores opciones encontradas.

## Tools disponibles

{tools_docs}

## Ejemplo de flujo

Thought: El usuario quiere volar de Madrid a Roma el 15 de junio. Primero buscaré vuelos.
Action: search_flights
Action Input: {{"origin": "MAD", "destination": "FCO", "date": "2025-06-15", "passengers": 1}}
Observation: [resultado de la búsqueda de vuelos]
Thought: Tengo los vuelos. Ahora busco hotel en Roma para esas fechas.
Action: search_hotels
Action Input: {{"city": "Roma", "check_in": "2025-06-15", "check_out": "2025-06-20", "guests": 1}}
Observation: [resultado de hoteles]
Thought: Ahora busco transporte del aeropuerto FCO al hotel seleccionado.
Action: search_airport_transport
Action Input: {{"airport": "FCO", "destination": "Hotel Colosseo", "datetime": "2025-06-15 14:30"}}
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


def parse_react_response(text: str) -> ReActStep:
    """
    Parsea la salida del LLM buscando los bloques Thought / Action / Action Input
    o Final Answer.

    El parser es tolerante a variaciones menores de formato (espacios, mayúsculas).
    """
    step = ReActStep()

    # Thought
    thought_match = re.search(r"Thought\s*:\s*(.+?)(?=Action\s*:|Final Answer\s*:|$)", text, re.S | re.I)
    if thought_match:
        step.thought = thought_match.group(1).strip()

    # Final Answer (tiene prioridad sobre Action si ambos aparecen)
    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.S | re.I)
    if final_match:
        step.final_answer = final_match.group(1).strip()
        return step

    # Action
    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.I)
    if action_match:
        step.action = action_match.group(1).strip()

    # Action Input — extrae el primer bloque JSON válido
    input_match = re.search(r"Action Input\s*:\s*(\{.*?\})", text, re.S | re.I)
    if input_match:
        raw_json = input_match.group(1).strip()
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
    max_iterations: int = 10
    temperature: float = 0.2
    max_new_tokens: int = 512

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
            logger.info(f"Thought: {step.thought[:120]}...")

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
                self._messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
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