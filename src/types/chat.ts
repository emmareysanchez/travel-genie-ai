export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface TraceEvent {
  type: "status" | "thought" | "action" | "observation" | "warning" | "error" | "final_answer";
  content: string;
}

export interface ChatResponse {
  assistant_message: string;
  trace: TraceEvent[];
}
