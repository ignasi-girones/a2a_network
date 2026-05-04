/**
 * TableDecorations — los objetos decorativos sobre la mesa para que
 * parezca una reunión real: laptops, papeles, libreta, taza, lápices.
 *
 * Vista cenital (desde arriba). Coordenadas relativas al centro
 * de la mesa en (400, 360).
 */

export function TableDecorations() {
  return (
    <g>
      {/* ── LAPTOP delante de AE1 (abajo-izquierda) ─────────────── */}
      <g transform="translate(280, 415)">
        {/* Sombra del laptop */}
        <rect x="-32" y="-22" width="64" height="44" rx="2" fill="rgba(15,23,42,0.2)" transform="translate(2, 3)" />
        {/* Cuerpo del laptop */}
        <rect x="-32" y="-22" width="64" height="44" rx="2" fill="#475569" stroke="#1e293b" strokeWidth="1.5" />
        {/* Pantalla (vista desde arriba, casi cerrada) */}
        <rect x="-30" y="-20" width="60" height="40" rx="1" fill="#0f172a" />
        {/* Reflejo de la pantalla */}
        <rect x="-26" y="-16" width="20" height="6" rx="1" fill="#3b82f6" opacity="0.6" />
        <rect x="-26" y="-8" width="40" height="2" rx="0.5" fill="#64748b" opacity="0.5" />
        <rect x="-26" y="-4" width="35" height="2" rx="0.5" fill="#64748b" opacity="0.5" />
        <rect x="-26" y="0" width="38" height="2" rx="0.5" fill="#64748b" opacity="0.5" />
        {/* Logo Apple-like en la tapa */}
        <circle cx="0" cy="11" r="3" fill="#94a3b8" opacity="0.5" />
      </g>

      {/* ── LAPTOP delante de AE2 (abajo-derecha) ───────────────── */}
      <g transform="translate(520, 415)">
        <rect x="-32" y="-22" width="64" height="44" rx="2" fill="rgba(15,23,42,0.2)" transform="translate(2, 3)" />
        <rect x="-32" y="-22" width="64" height="44" rx="2" fill="#831843" stroke="#1e293b" strokeWidth="1.5" />
        <rect x="-30" y="-20" width="60" height="40" rx="1" fill="#0f172a" />
        {/* Lineas de código */}
        <rect x="-26" y="-16" width="14" height="2" rx="0.5" fill="#10b981" opacity="0.7" />
        <rect x="-10" y="-16" width="20" height="2" rx="0.5" fill="#fbbf24" opacity="0.7" />
        <rect x="-26" y="-12" width="30" height="2" rx="0.5" fill="#94a3b8" opacity="0.5" />
        <rect x="-22" y="-8" width="25" height="2" rx="0.5" fill="#94a3b8" opacity="0.5" />
        <rect x="-22" y="-4" width="32" height="2" rx="0.5" fill="#94a3b8" opacity="0.5" />
        <rect x="-26" y="0" width="20" height="2" rx="0.5" fill="#ec4899" opacity="0.7" />
      </g>

      {/* ── LIBRETA + LÁPIZ delante de AE3 (arriba) ─────────────── */}
      <g transform="translate(400, 270)">
        {/* Sombra */}
        <rect x="-30" y="-22" width="60" height="44" rx="2" fill="rgba(15,23,42,0.18)" transform="translate(2, 3)" />
        {/* Libreta */}
        <rect x="-30" y="-22" width="60" height="44" rx="2" fill="#fef3c7" stroke="#92400e" strokeWidth="1.5" />
        {/* Espiral de la libreta */}
        <line x1="-30" y1="-22" x2="-30" y2="22" stroke="#78350f" strokeWidth="2" />
        <circle cx="-30" cy="-15" r="1.5" fill="#78350f" />
        <circle cx="-30" cy="-7" r="1.5" fill="#78350f" />
        <circle cx="-30" cy="1" r="1.5" fill="#78350f" />
        <circle cx="-30" cy="9" r="1.5" fill="#78350f" />
        <circle cx="-30" cy="17" r="1.5" fill="#78350f" />
        {/* Líneas escritas */}
        <line x1="-22" y1="-14" x2="22" y2="-14" stroke="#3b82f6" strokeWidth="1" opacity="0.6" />
        <line x1="-22" y1="-8" x2="18" y2="-8" stroke="#3b82f6" strokeWidth="1" opacity="0.6" />
        <line x1="-22" y1="-2" x2="22" y2="-2" stroke="#3b82f6" strokeWidth="1" opacity="0.6" />
        <line x1="-22" y1="4" x2="14" y2="4" stroke="#3b82f6" strokeWidth="1" opacity="0.6" />
        <line x1="-22" y1="10" x2="20" y2="10" stroke="#3b82f6" strokeWidth="1" opacity="0.6" />
        <line x1="-22" y1="16" x2="10" y2="16" stroke="#3b82f6" strokeWidth="1" opacity="0.6" />

        {/* Lápiz al lado de la libreta */}
        <g transform="translate(40, 0) rotate(15)">
          <rect x="-4" y="-25" width="8" height="40" fill="#fbbf24" stroke="#92400e" strokeWidth="1" />
          <polygon points="-4,15 4,15 0,22" fill="#fef3c7" stroke="#92400e" strokeWidth="1" />
          <polygon points="-2,18 2,18 0,22" fill="#1e293b" />
          <rect x="-4" y="-30" width="8" height="6" fill="#dc2626" stroke="#991b1b" strokeWidth="1" />
          <rect x="-4" y="-32" width="8" height="3" fill="#94a3b8" />
        </g>
      </g>

      {/* ── TAZA DE CAFÉ a la izquierda ─────────────────────────── */}
      <g transform="translate(335, 360)">
        {/* Asa */}
        <ellipse cx="14" cy="0" rx="6" ry="9" fill="none" stroke="#78350f" strokeWidth="2.5" />
        {/* Sombra */}
        <circle cx="0" cy="0" r="14" fill="rgba(15,23,42,0.2)" transform="translate(2, 3)" />
        {/* Taza vista desde arriba */}
        <circle cx="0" cy="0" r="14" fill="#ffffff" stroke="#1e293b" strokeWidth="2" />
        {/* Café dentro */}
        <circle cx="0" cy="0" r="11" fill="#5d3a1a" />
        {/* Espuma */}
        <ellipse cx="-3" cy="-2" rx="5" ry="3" fill="#a78b6f" opacity="0.7" />
        {/* Vapor (líneas onduladas) */}
        <path d="M -4 -16 Q -6 -20, -4 -24 Q -2 -28, -4 -32" stroke="#cbd5e1" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.7" />
        <path d="M 0 -16 Q 2 -20, 0 -24 Q -2 -28, 0 -32" stroke="#cbd5e1" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.7" />
        <path d="M 4 -16 Q 6 -20, 4 -24 Q 2 -28, 4 -32" stroke="#cbd5e1" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.7" />
      </g>

      {/* ── PAPELES SUELTOS a la derecha ────────────────────────── */}
      <g transform="translate(465, 350) rotate(-12)">
        <rect x="-18" y="-22" width="36" height="44" rx="1" fill="rgba(15,23,42,0.15)" transform="translate(2, 3)" />
        <rect x="-18" y="-22" width="36" height="44" rx="1" fill="white" stroke="#94a3b8" strokeWidth="1" />
        <line x1="-12" y1="-15" x2="12" y2="-15" stroke="#64748b" strokeWidth="0.8" />
        <line x1="-12" y1="-10" x2="10" y2="-10" stroke="#64748b" strokeWidth="0.8" />
        <line x1="-12" y1="-5" x2="12" y2="-5" stroke="#64748b" strokeWidth="0.8" />
        <line x1="-12" y1="0" x2="8" y2="0" stroke="#64748b" strokeWidth="0.8" />
        {/* Pequeño gráfico circular */}
        <circle cx="0" cy="12" r="6" fill="none" stroke="#3b82f6" strokeWidth="1.5" />
        <path d="M 0 12 L 0 6 A 6 6 0 0 1 5 14 Z" fill="#3b82f6" opacity="0.6" />
      </g>

      <g transform="translate(478, 372) rotate(8)">
        <rect x="-15" y="-18" width="30" height="36" rx="1" fill="white" stroke="#94a3b8" strokeWidth="1" />
        <line x1="-10" y1="-12" x2="10" y2="-12" stroke="#64748b" strokeWidth="0.8" />
        <line x1="-10" y1="-8" x2="8" y2="-8" stroke="#64748b" strokeWidth="0.8" />
        <line x1="-10" y1="-4" x2="10" y2="-4" stroke="#64748b" strokeWidth="0.8" />
        <line x1="-10" y1="0" x2="6" y2="0" stroke="#64748b" strokeWidth="0.8" />
        {/* Mini gráfico de barras */}
        <rect x="-8" y="6" width="3" height="8" fill="#10b981" />
        <rect x="-3" y="3" width="3" height="11" fill="#3b82f6" />
        <rect x="2" y="8" width="3" height="6" fill="#f59e0b" />
        <rect x="7" y="5" width="3" height="9" fill="#ec4899" />
      </g>
    </g>
  );
}
