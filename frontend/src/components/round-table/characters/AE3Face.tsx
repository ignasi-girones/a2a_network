/**
 * AE3 visto desde arriba.
 * Personaje: chica rubia con moño/recogido, gafas, jersey morado.
 */

interface Props {
  accent?: string;
  secondary?: string;
}

export function AE3Face({ accent = '#c026d3', secondary = '#7e22ce' }: Props) {
  return (
    <g>
      {/* Sombra del cuerpo */}
      <ellipse cx="100" cy="200" rx="75" ry="22" fill="rgba(15,23,42,0.18)" />

      {/* Brazos */}
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
      <ellipse cx="84" cy="212" rx="9" ry="11" fill="#fbe2c5" stroke="#5a3a2a" strokeWidth="1.5" />
      <ellipse cx="116" cy="212" rx="9" ry="11" fill="#fbe2c5" stroke="#5a3a2a" strokeWidth="1.5" />

      {/* Torso (jersey morado) */}
      <path
        d="M 60 175 Q 60 130, 100 130 Q 140 130, 140 175 L 140 195 Q 100 210, 60 195 Z"
        fill={accent}
        stroke={secondary}
        strokeWidth="2.5"
      />
      <ellipse cx="100" cy="135" rx="14" ry="6" fill={secondary} />

      {/* Cabeza */}
      <circle cx="100" cy="90" r="42" fill="#fbe2c5" stroke="#5a3a2a" strokeWidth="2.5" />

      {/* Pelo rubio (peinado hacia atrás) */}
      <path
        d="M 58 92 Q 58 50, 100 48 Q 142 50, 142 92 Q 140 80, 132 72 Q 116 70, 100 72 Q 84 70, 68 72 Q 60 80, 58 92 Z"
        fill="#fbbf24"
      />
      {/* Reflejos dorados */}
      <path d="M 70 70 Q 80 65, 90 68" stroke="#fde68a" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M 110 68 Q 120 65, 130 70" stroke="#fde68a" strokeWidth="2" fill="none" strokeLinecap="round" />

      {/* MOÑO en lo alto (visto desde arriba se ve grande) */}
      <circle cx="100" cy="55" r="14" fill="#f59e0b" stroke="#92400e" strokeWidth="2" />
      <circle cx="96" cy="51" r="3" fill="#fde68a" opacity="0.6" />
      {/* Goma del moño */}
      <ellipse cx="100" cy="68" rx="10" ry="3" fill={accent} />

      {/* Frente despejada */}
      <path d="M 78 95 Q 100 92, 122 95 Q 122 100, 100 102 Q 78 100, 78 95 Z" fill="#fbe2c5" opacity="0.6" />

      {/* GAFAS (rasgo distintivo de AE3) */}
      <circle cx="86" cy="106" r="6.5" fill="white" stroke="#1e293b" strokeWidth="1.8" opacity="0.85" />
      <circle cx="114" cy="106" r="6.5" fill="white" stroke="#1e293b" strokeWidth="1.8" opacity="0.85" />
      <line x1="92.5" y1="106" x2="107.5" y2="106" stroke="#1e293b" strokeWidth="1.8" />
      {/* Pupilas dentro de las gafas */}
      <circle cx="86" cy="106" r="2" fill="#1e293b" />
      <circle cx="114" cy="106" r="2" fill="#1e293b" />

      {/* Nariz */}
      <ellipse cx="100" cy="118" rx="2.5" ry="3.5" fill="#5a3a2a" opacity="0.3" />

      {/* Mejillas suaves */}
      <circle cx="76" cy="118" r="3.5" fill="#ec4899" opacity="0.25" />
      <circle cx="124" cy="118" r="3.5" fill="#ec4899" opacity="0.25" />

      {/* Orejas */}
      <ellipse cx="58" cy="100" rx="4" ry="7" fill="#f0c9a4" stroke="#5a3a2a" strokeWidth="1" />
      <ellipse cx="142" cy="100" rx="4" ry="7" fill="#f0c9a4" stroke="#5a3a2a" strokeWidth="1" />
    </g>
  );
}
