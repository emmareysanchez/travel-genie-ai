import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronRight,
  ChevronLeft,
  Activity,
  Sparkles,
  Brain,
  Wrench,
  Search,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import type { TraceEvent } from "@/types/chat";

interface TripInsightsPanelProps {
  isOpen: boolean;
  onToggle: () => void;
  trace: TraceEvent[];
}

const traceConfig: Record<TraceEvent["type"], { icon: React.ElementType; label: string; color: string }> = {
  status: { icon: Sparkles, label: "Status", color: "text-primary" },
  thought: { icon: Brain, label: "Thought", color: "text-purple-400" },
  action: { icon: Wrench, label: "Action", color: "text-blue-400" },
  observation: { icon: Search, label: "Observation", color: "text-emerald-400" },
  warning: { icon: AlertTriangle, label: "Warning", color: "text-yellow-400" },
  error: { icon: AlertCircle, label: "Error", color: "text-destructive" },
  final_answer: { icon: CheckCircle2, label: "Final Answer", color: "text-green-400" },
};

const TripInsightsPanel = ({ isOpen, onToggle, trace }: TripInsightsPanelProps) => {
  return (
    <>
      {/* Toggle button */}
      <button
        onClick={onToggle}
        className="fixed right-0 top-1/2 -translate-y-1/2 z-30 glass-panel rounded-l-lg p-2 hover:border-primary/40 transition-colors"
        title={isOpen ? "Close panel" : "Agent activity"}
      >
        {isOpen ? (
          <ChevronRight className="w-4 h-4 text-muted-foreground" />
        ) : (
          <div className="flex flex-col items-center gap-1">
            <Activity className="w-4 h-4 text-primary" />
            <ChevronLeft className="w-3 h-3 text-muted-foreground" />
          </div>
        )}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.aside
            initial={{ x: "100%", opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-80 z-20 bg-card/90 backdrop-blur-2xl border-l border-border flex flex-col"
          >
            <div className="mt-16 px-5 pb-6 flex flex-col flex-1 min-h-0">
              <h2 className="font-display text-lg font-semibold text-foreground mb-1">
                Agent Activity
              </h2>
              <p className="text-xs text-muted-foreground mb-5">
                Real-time reasoning trace
              </p>

              <div className="flex-1 overflow-y-auto space-y-1 pr-1 scrollbar-thin">
                {trace.length === 0 ? (
                  <p className="text-xs text-muted-foreground italic py-4 text-center">
                    No activity yet — send a message to see the agent's reasoning.
                  </p>
                ) : (
                  trace.map((event, i) => {
                    const cfg = traceConfig[event.type] || traceConfig.status;
                    const Icon = cfg.icon;
                    return (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, x: 16 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.03 }}
                        className="flex gap-3 py-2.5 px-3 rounded-lg hover:bg-secondary/40 transition-colors"
                      >
                        <div className="flex flex-col items-center pt-0.5">
                          <Icon className={`w-4 h-4 flex-shrink-0 ${cfg.color}`} />
                          {i < trace.length - 1 && (
                            <div className="w-px flex-1 bg-border/60 mt-1" />
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <span className={`text-[10px] font-semibold uppercase tracking-wider ${cfg.color}`}>
                            {cfg.label}
                          </span>
                          <p className="text-xs text-foreground/80 leading-relaxed mt-0.5 break-words whitespace-pre-wrap">
                            {event.content}
                          </p>
                        </div>
                      </motion.div>
                    );
                  })
                )}
              </div>
            </div>

            <div className="px-5 py-4 border-t border-border">
              <p className="text-[10px] text-muted-foreground text-center">
                Powered by Travel Genie AI ✨
              </p>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
};

export default TripInsightsPanel;
