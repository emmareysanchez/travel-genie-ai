import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { Sparkles, User } from "lucide-react";

interface ChatBubbleProps {
  role: "user" | "assistant";
  content: string;
}

const ChatBubble = ({ role, content }: ChatBubbleProps) => {
  const isUser = role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      <div
        className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center shadow-lg ${
          isUser ? "bg-primary glow-sm" : "glass-panel"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-primary-foreground" />
        ) : (
          <Sparkles className="w-4 h-4 text-primary" />
        )}
      </div>
      <div
        className={`max-w-[78%] rounded-2xl px-5 py-3.5 shadow-md ${
          isUser
            ? "bg-bubble-user text-bubble-user-foreground rounded-br-md"
            : "bg-bubble-assistant border border-bubble-assistant-border text-foreground rounded-bl-md"
        }`}
      >
        <div className="prose prose-sm prose-invert max-w-none text-inherit leading-relaxed [&>p]:m-0 [&>ul]:mt-2 [&>ol]:mt-2">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
      </div>
    </motion.div>
  );
};

export default ChatBubble;
