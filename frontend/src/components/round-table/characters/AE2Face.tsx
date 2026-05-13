/**
 * AE2 visto desde arriba.
 * Personaje: chica pelirroja, coleta visible, jersey verde.
 */

interface Props {
  accent?: string;
  secondary?: string;
}

export function AE2Face({ accent = '#059669', secondary = '#047857' }: Props) {
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
      <ellipse cx="84" cy="212" rx="9" ry="11" fill="#f5d8b8" stroke="#5a3a2a" strokeWidth="1.5" />
      <ellipse cx="116" cy="212" rx="9" ry="11" fill="#f5d8b8" stroke="#5a3a2a" strokeWidth="1.5" />

      {/* Torso (jersey verde) */}
      <path
        d="M 60 175 Q 60 130, 100 130 Q 140 130, 140 175 L 140 195 Q 100 210, 60 195 Z"
        fill={accent}
        stroke={secondary}
        strokeWidth="2.5"
      />
      <ellipse cx="100" cy="135" rx="14" ry="6" fill={secondary} />

      {/* Coleta saliendo por detrás de la cabeza */}
      <ellipse
        cx="100"
        cy="148"
        rx="14"
        ry="22"
        fill="#c2410c"
        stroke="#7c2d12"
        strokeWidth="1.5"
      />
      <path
        d="M 92 165 Q 95 175, 100 178 Q 105 175, 108 165"
        stroke="#7c2d12"
        strokeWidth="1"
        fill="none"
      />

      {/* Cabeza */}
      <circle cx="100" cy="90" r="42" fill="#f5d8b8" stroke="#5a3a2a" strokeWidth="2.5" />

      {/* Pelo pelirrojo, ondulado (vista cenital) */}
      <path
        d="M 58 92 Q 56 50, 100 46 Q 144 50, 142 92 Q 144 78, 138 68 Q 130 60, 120 62 Q 110 56, 100 58 Q 90 56, 80 62 Q 70 60, 62 68 Q 56 78, 58 92 Z"
        fill="#c2410c"
      />
      {/* Reflejos del pelo */}
      <path d="M 75 65 Q 85 58, 95 62" stroke="#ea580c" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M 105 62 Q 115 58, 125 65" stroke="#ea580c" strokeWidth="2" fill="none" strokeLinecap="round" />
      {/* Raya en medio */}
      <path d="M 100 50 L 100 75" stroke="#7c2d12" strokeWidth="1.5" />

      {/* Frente despejada */}
      <path d="M 78 95 Q 100 92, 122 95 Q 122 100, 100 102 Q 78 100, 78 95 Z" fill="#f5d8b8" opacity="0.6" />

      {/* Ojos con pestañas */}
      <ellipse cx="86" cy="105" rx="3" ry="2.2" fill="#1e293b" />
      <ellipse cx="114" cy="105" rx="3" ry="2.2" fill="#1e293b" />
      {/* Pestañitas */}
      <path d="M 82 102 L 80 100 M 90 102 L 92 100" stroke="#1e293b" strokeWidth="1" strokeLinecap="round" />
      <path d="M 110 102 L 108 100 M 118 102 L 120 100" stroke="#1e293b" strokeWidth="1" strokeLinecap="round" />

      {/* Nariz */}
      <ellipse cx="100" cy="115" rx="2.5" ry="3.5" fill="#5a3a2a" opacity="0.3" />

      {/* Mejillas sonrosadas */}
      <circle cx="78" cy="115" r="4" fill="#ec4899" opacity="0.3" />
      <circle cx="122" cy="115" r="4" fill="#ec4899" opacity="0.3" />

      {/* Orejas con pendiente */}
      <ellipse cx="58" cy="100" rx="4" ry="7" fill="#f0c9a4" stroke="#5a3a2a" strokeWidth="1" />
      <ellipse cx="142" cy="100" rx="4" ry="7" fill="#f0c9a4" stroke="#5a3a2a" strokeWidth="1" />
      <circle cx="58" cy="106" r="1.5" fill={accent} />
      <circle cx="142" cy="106" r="1.5" fill={accent} />
    </g>
  );
}
