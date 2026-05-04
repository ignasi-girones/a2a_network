/**
 * AE1 visto desde arriba (vista cenital).
 *
 * Estilo cartoon plano, similar a las ilustraciones de Freepik.
 * Personaje: chico moreno, pelo corto y oscuro, jersey azul.
 *
 * El SVG está pensado para un viewBox interno de 200x240 donde:
 *   - (100, 60)  es el centro de la cabeza
 *   - (100, 180) es el centro del cuerpo (hombros + brazos sobre la mesa)
 *
 * El personaje "mira" hacia arriba (hacia el norte). El componente
 * AgentCharacter se encarga de rotarlo según la posición que ocupa.
 */

interface Props {
  accent?: string;
  secondary?: string;
}

export function AE1Face({ accent = '#2563eb', secondary = '#1e40af' }: Props) {
  return (
    <g>
      {/* Sombra del cuerpo en el suelo */}
      <ellipse cx="100" cy="200" rx="75" ry="22" fill="rgba(15,23,42,0.18)" />

      {/* Brazos extendidos hacia adelante (hacia la mesa) */}
      <path
        d="M 55 175 Q 60 200, 78 215 L 95 200 Q 80 185, 70 165 Z"
        fill={accent}
        stroke={secondary}
        strokeWidth="2"
      />
      <path
        d="M 145 175 Q 140 200, 122 215 L 105 200 Q 120 185, 130 165 Z"
        fill={accent}
        stroke={secondary}
        strokeWidth="2"
      />

      {/* Manos */}
      <ellipse cx="84" cy="212" rx="9" ry="11" fill="#fde7d3" stroke="#5a3a2a" strokeWidth="1.5" />
      <ellipse cx="116" cy="212" rx="9" ry="11" fill="#fde7d3" stroke="#5a3a2a" strokeWidth="1.5" />

      {/* Torso (jersey azul, hombros visibles desde arriba) */}
      <path
        d="M 60 175 Q 60 130, 100 130 Q 140 130, 140 175 L 140 195 Q 100 210, 60 195 Z"
        fill={accent}
        stroke={secondary}
        strokeWidth="2.5"
      />

      {/* Detalle del cuello del jersey */}
      <ellipse cx="100" cy="135" rx="14" ry="6" fill={secondary} />

      {/* Cabeza (vista desde arriba) */}
      <circle cx="100" cy="90" r="42" fill="#fde7d3" stroke="#5a3a2a" strokeWidth="2.5" />

      {/* Pelo (cubre la mayor parte de la cabeza vista desde arriba) */}
      <path
        d="M 60 92 Q 60 50, 100 48 Q 140 50, 140 92 Q 138 78, 130 70 Q 115 65, 100 68 Q 85 65, 70 70 Q 62 78, 60 92 Z"
        fill="#3a2a1f"
      />
      {/* Mechón decorativo */}
      <path d="M 88 60 Q 100 52, 112 60 Q 105 70, 100 68 Q 95 70, 88 60 Z" fill="#1a1208" />

      {/* Frente despejada */}
      <path d="M 75 95 Q 100 90, 125 95 Q 125 100, 100 102 Q 75 100, 75 95 Z" fill="#fde7d3" opacity="0.6" />

      {/* Ojos cerrados/concentrados (visto desde arriba apenas se ven, dos puntitos) */}
      <ellipse cx="86" cy="105" rx="2.5" ry="2" fill="#1e293b" />
      <ellipse cx="114" cy="105" rx="2.5" ry="2" fill="#1e293b" />

      {/* Nariz pequeña (sombra desde arriba) */}
      <ellipse cx="100" cy="115" rx="2.5" ry="3.5" fill="#5a3a2a" opacity="0.3" />

      {/* Orejas (desde arriba se ven a los lados) */}
      <ellipse cx="60" cy="100" rx="4" ry="7" fill="#f0c9a4" stroke="#5a3a2a" strokeWidth="1" />
      <ellipse cx="140" cy="100" rx="4" ry="7" fill="#f0c9a4" stroke="#5a3a2a" strokeWidth="1" />
    </g>
  );
}
