# Travel Genie AI

An AI-powered travel planning assistant built with a ReAct (Reasoning + Acting) agent architecture. The system engages users in conversation to gather trip requirements and uses multiple real-time APIs to search for flights, hotels, transport routes, points of interest, and food recommendations — delivering personalized, structured travel plans.

---

## Project Structure

```
travel-genie-ai/
├── backend/                  # ✅ Final version of the project
│   ├── tools/                #    External API integrations
│   ├── travel_agent.py       #    Core ReAct agent
│   └── api.py                #    FastAPI server with SSE streaming
├── agent/                    # Earlier/alternative agent implementations (local use)
├── src/                      # React/TypeScript frontend
│   ├── components/           #    UI components (chat, trip insights, suggestions)
│   ├── pages/                #    Page-level components
│   └── types/                #    TypeScript type definitions
├── data/                     # Mock data for testing
├── tools/                    # Additional tool scripts (local use)
├── package.json              # Node.js dependencies
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Python project configuration
```

> **Note:** The **final version** of the backend is located in the `backend/` folder. Files placed at the root level (e.g., `agent/`, `tools/`) are earlier iterations used for local development and testing.

---

## Architecture Overview

The application follows a client–server architecture:

1. **React Frontend** — Chat interface with Server-Sent Events (SSE) streaming, agent reasoning trace display, and a trip insights panel.
2. **FastAPI Backend** — ReAct agent powered by a local LLM. Handles conversation state, tool dispatch, and streaming responses.
3. **Tool Integration Layer** — Six modular tools connecting to external APIs for flights, hotels, transport, geocoding, points of interest, and food.
4. **Evaluation Framework** — Offline test suite with completeness, relevance, budget adherence, and LLM-as-judge metrics.

### Agent Tools

| Tool | API | Purpose |
|------|-----|---------|
| `flights` | SerpAPI (Google Flights) | Search for available flights |
| `hotels` | Booking.com via RapidAPI | Search for hotel availability |
| `transport` | Mapbox / Google Maps | Calculate routes and travel times |
| `places_of_interest` | Geoapify | Find POIs and attractions |
| `food` | TasteAtlas | Local food and restaurant recommendations |
| `airports` | API Ninjas | Resolve airport codes |

---

## Tech Stack

**Frontend**
- React 18 + TypeScript
- Vite + SWC
- Tailwind CSS + shadcn/ui + Radix UI
- Framer Motion, React Markdown, Recharts

**Backend**
- Python 3.12
- FastAPI + Uvicorn
- PyTorch + Hugging Face Transformers (local LLM)
- Pydantic, python-dotenv

---

## Deployment & Evaluation

> **This project requires a GPU server and active API keys to run.**

The system runs a local LLM (Gemma) and connects to several paid/rate-limited external APIs. For this reason, **the deployment is not left permanently active**.

If you need to evaluate or test the project, **please contact us** and we will bring the deployment back online (via ngrok).

We will provide you with the active endpoint URL so you can interact with the application without needing to set up the infrastructure yourself.

---

## Local Setup (Reference)

If you want to run the project locally, you will need:

### Backend

```bash
cd backend
pip install -r ../requirements.txt
# Create a .env file with the required API keys (see below)
uvicorn api:app --host 0.0.0.0 --port 18007
```

### Frontend

```bash
npm install
npm run dev
```

The frontend dev server runs on `http://localhost:8080` and expects the backend at `http://127.0.0.1:18007`.

### Required Environment Variables

Create a `.env` file inside `backend/` with the following keys:

```env
SERPAPI_API_KEY=
RAPIDAPI_KEY=
MAPBOX_ACCESS_TOKEN=
GOOGLE_MAPS_API_KEY=
NINJA_API_KEY=
GEOAPIFY_API_KEY=
VITE_API_BASE_URL=http://127.0.0.1:18007
```

---

## Evaluation Framework

The project includes an evaluation suite that measures:

- **Efficiency:** latency, token usage, number of ReAct iterations
- **Objective quality:** completeness, geographic relevance, budget adherence
- **Semantic quality:** LLM-as-judge scoring

Results are saved to `resultados_eval.json`. See the project report (`InformeTravelGenieAI.pdf`) for full methodology and results.
