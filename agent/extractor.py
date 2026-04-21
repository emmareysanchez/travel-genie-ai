import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from agent.schemas import ChatMessage, ExtractedTrip


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
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


class TripExtractor:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens: int = 180,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if not torch.cuda.is_available():
            self.model.to("cpu")

        self.model.eval()

    def _build_prompt(self, history: list[ChatMessage], message: str) -> str:
        history_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in history
        ).strip()

        return f"""
Eres un extractor de información para una app de viajes.

Devuelve SOLO JSON válido con estos campos:
- origin
- destination
- departure_date
- return_date
- travelers
- interests
- budget

Reglas:
- Si un campo no está claro, usa null.
- travelers debe ser entero. Si no se dice, usa 1.
- interests debe ser una lista de strings.
- Convierte fechas a YYYY-MM-DD si es posible.
- No inventes datos.
- Responde solo con JSON.

Conversación previa:
{history_text if history_text else "(vacía)"}

Último mensaje del usuario:
{message}

Formato:
{{
  "origin": null,
  "destination": null,
  "departure_date": null,
  "return_date": null,
  "travelers": 1,
  "interests": [],
  "budget": null
}}
""".strip()

    def extract(self, history: list[ChatMessage], message: str) -> ExtractedTrip:
        prompt = self._build_prompt(history, message)

        messages = [
            {"role": "system", "content": "Devuelve únicamente JSON válido."},
            {"role": "user", "content": prompt},
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_device = next(self.model.parameters()).device
        enc = self.tokenizer(text_input, return_tensors="pt").to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen = outputs[0][enc["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        raw_json = _extract_balanced_json(response)
        if not raw_json:
            raise ValueError(f"No se pudo extraer JSON del modelo. Respuesta: {response}")

        data = json.loads(raw_json)

        return ExtractedTrip(
            origin=data.get("origin"),
            destination=data.get("destination"),
            departure_date=data.get("departure_date"),
            return_date=data.get("return_date"),
            travelers=data.get("travelers") or 1,
            interests=data.get("interests") or [],
            budget=data.get("budget"),
        )