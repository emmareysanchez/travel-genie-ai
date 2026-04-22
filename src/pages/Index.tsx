import { useState, useRef, useEffect } from "react";
import { Send, RotateCcw, Loader2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ChatBubble from "@/components/ChatBubble";
import SuggestedPrompts from "@/components/SuggestedPrompts";
import TripInsightsPanel from "@/components/TripInsightsPanel";
import genieLamp from "@/assets/genie-lamp.png";
import type { Message, TraceEvent } from "@/types/chat";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";
const buildApiUrl = (path: string) => `${API_BASE_URL}${path}`;

const WELCOME_MESSAGE =
  "✨ Welcome, traveler! I'm your **Travel Genie** — your personal guide to discovering the world's most breathtaking destinations.\n\nAsk me anything about flights, hotels, itineraries, or hidden gems!";

async function consumeSSEStream(
  response: Response,
  onEvent: (event: { type: string; content: string }) => void
) {
  if (!response.body) {
    throw new Error("Streaming response has no body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const lines = chunk
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);

      const dataLines = lines
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.replace(/^data:\s?/, ""));

      if (dataLines.length === 0) continue;

      const raw = dataLines.join("\n");

      try {
        const parsed = JSON.parse(raw);
        onEvent(parsed);
      } catch (error) {
        console.error("Error parsing SSE event:", raw, error);
      }
    }
  }

  if (buffer.trim()) {
    const lines = buffer
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    const dataLines = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.replace(/^data:\s?/, ""));

    if (dataLines.length > 0) {
      const raw = dataLines.join("\n");
      try {
        const parsed = JSON.parse(raw);
        onEvent(parsed);
      } catch (error) {
        console.error("Error parsing last SSE event:", raw, error);
      }
    }
  }
}

const Index = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: WELCOME_MESSAGE },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [insightsOpen, setInsightsOpen] = useState(false);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const showSuggestions = messages.length === 1;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  if (!API_BASE_URL) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="glass-panel rounded-2xl p-8 max-w-md text-center space-y-3">
          <AlertCircle className="w-10 h-10 text-destructive mx-auto" />
          <h2 className="font-display text-lg font-semibold text-foreground">
            Backend not configured
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Set the{" "}
            <code className="text-primary font-mono text-xs bg-secondary/60 px-1.5 py-0.5 rounded">
              VITE_API_BASE_URL
            </code>{" "}
            environment variable to your backend URL and reload.
          </p>
        </div>
      </div>
    );
  }

  const sendMessage = async (text?: string) => {
    const trimmed = (text ?? input).trim();
    if (!trimmed || isLoading) return;

    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setInput("");
    setIsLoading(true);
    setTrace([]);
    setInsightsOpen(true);

    try {
      const res = await fetch(buildApiUrl("/chat/stream"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ message: trimmed, reset: false }),
      });

      if (!res.ok) {
        throw new Error(`Server error (${res.status})`);
      }

      let finalAnswer = "";

      await consumeSSEStream(res, (event) => {
        if (event.type === "done") return;

        if (event.type === "error") {
          setError(event.content || "Something went wrong. Please try again.");
          setTrace((prev) => [...prev, event as TraceEvent]);
          return;
        }

        if (event.type === "final_answer") {
          finalAnswer = event.content;
          setTrace((prev) => [...prev, event as TraceEvent]);
          return;
        }

        setTrace((prev) => [...prev, event as TraceEvent]);
      });

      if (finalAnswer) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: finalAnswer },
        ]);
      }
    } catch (err: any) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const resetConversation = async () => {
    try {
      await fetch(buildApiUrl("/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ message: "", reset: true }),
      });
    } catch {}

    setMessages([{ role: "assistant", content: WELCOME_MESSAGE }]);
    setTrace([]);
    setError(null);
    setInput("");
    setInsightsOpen(false);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background relative overflow-hidden">
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/[0.03] blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[40%] h-[40%] rounded-full bg-primary/[0.04] blur-[100px]" />
      </div>

      <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-border/60 glass-panel">
        <div className="flex items-center gap-3.5">
          <motion.img
            src={genieLamp}
            alt="Travel Genie"
            width={38}
            height={38}
            className="drop-shadow-lg"
            initial={{ rotate: -10 }}
            animate={{ rotate: 0 }}
            transition={{ type: "spring", stiffness: 200 }}
          />
          <div>
            <h1 className="font-display text-xl font-semibold leading-tight">
              <span className="text-gradient-gold">Travel Genie</span>
            </h1>
            <p className="text-[11px] text-muted-foreground tracking-wide">
              Your AI travel companion
            </p>
          </div>
        </div>
        <button
          onClick={resetConversation}
          className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground glass-panel hover:border-primary/30 transition-all duration-200"
          title="Reset conversation"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">New chat</span>
        </button>
      </header>

      <main className="flex-1 overflow-y-auto px-4 sm:px-6 py-8 relative z-10">
        <div className="max-w-2xl mx-auto space-y-6">
          <AnimatePresence>
            {messages.map((msg, i) => (
              <ChatBubble key={i} role={msg.role} content={msg.content} />
            ))}
          </AnimatePresence>

          {showSuggestions && (
            <SuggestedPrompts onSelect={(text) => sendMessage(text)} />
          )}

          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3"
            >
              <div className="w-9 h-9 rounded-full glass-panel flex items-center justify-center">
                <Loader2 className="w-4 h-4 text-primary animate-spin" />
              </div>
              <div className="bg-bubble-assistant border border-bubble-assistant-border rounded-2xl rounded-bl-md px-5 py-3.5">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse-glow" />
                  <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse-glow [animation-delay:0.3s]" />
                  <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse-glow [animation-delay:0.6s]" />
                </div>
              </div>
            </motion.div>
          )}

          {error && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2.5 text-destructive text-sm glass-panel border-destructive/30 rounded-xl px-4 py-3"
            >
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {error}
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      <div className="relative z-10 border-t border-border/60 glass-panel px-4 sm:px-6 py-4">
        <div className="max-w-2xl mx-auto flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about destinations, flights, hotels..."
            className="flex-1 bg-secondary/60 text-foreground placeholder:text-muted-foreground rounded-xl px-5 py-3.5 text-sm outline-none border border-border/50 focus:border-primary/50 focus:ring-2 focus:ring-primary/20 transition-all duration-200"
            disabled={isLoading}
          />
          <button
            onClick={() => sendMessage()}
            disabled={isLoading || !input.trim()}
            className="bg-primary text-primary-foreground rounded-xl px-4 py-3.5 hover:brightness-110 disabled:opacity-30 transition-all duration-200 flex-shrink-0 glow-sm"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      <TripInsightsPanel
        isOpen={insightsOpen}
        onToggle={() => setInsightsOpen(!insightsOpen)}
        trace={trace}
      />
    </div>
  );
};

export default Index;