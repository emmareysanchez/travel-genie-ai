import { motion } from "framer-motion";
import { Compass, MapPin, Plane, UtensilsCrossed } from "lucide-react";

interface SuggestedPromptsProps {
  onSelect: (prompt: string) => void;
}

const prompts = [
  {
    icon: Plane,
    text: "Quiero viajar de Madrid a Roma del 10 al 13 de junio de 2026 para 2 personas y me interesan museos y restaurantes",
    label: "Madrid → Roma",
  },
  {
    icon: MapPin,
    text: "Planea una escapada a París con hoteles y lugares culturales",
    label: "París cultural",
  },
  {
    icon: UtensilsCrossed,
    text: "Quiero un viaje gastronómico a Lisboa",
    label: "Lisboa gastronómica",
  },
  {
    icon: Compass,
    text: "Organízame un fin de semana en Berlín con vida nocturna",
    label: "Berlín nocturno",
  },
];

const SuggestedPrompts = ({ onSelect }: SuggestedPromptsProps) => {
  return (
    <div className="flex flex-col items-center gap-6 py-8">
      <p className="text-sm text-muted-foreground tracking-wide uppercase font-medium">
        Prueba a preguntar
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
        {prompts.map((p, i) => (
          <motion.button
            key={i}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + i * 0.1, duration: 0.4 }}
            onClick={() => onSelect(p.text)}
            className="group glass-panel rounded-xl px-4 py-3.5 flex items-start gap-3 hover:border-primary/40 hover:glow-sm transition-all duration-300 cursor-pointer text-left"
          >
            <p.icon className="w-4 h-4 text-primary opacity-70 group-hover:opacity-100 transition-opacity mt-0.5 flex-shrink-0" />
            <div>
              <span className="text-xs font-semibold text-primary/80 block mb-0.5">{p.label}</span>
              <span className="text-sm text-secondary-foreground group-hover:text-foreground transition-colors leading-snug">
                {p.text}
              </span>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

export default SuggestedPrompts;
