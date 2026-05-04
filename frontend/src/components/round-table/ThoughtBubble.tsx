/**
 * ThoughtBubble - bocadillo de comic limitado al lienzo de la escena.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';
import { AGENT_VISUAL, type AgentId } from '../../lib/round-table-config';

interface Props {
  agentId: AgentId;
  text: string;
  visible: boolean;
  charsPerSecond?: number;
}

const SCENE_WIDTH = 800;
const SCENE_PADDING = 20;

const BUBBLE_CONFIG: Record<
  AgentId,
  {
    width: number;
    height: number;
    y: number;
    xInset: number;
    anchorXOffset: number;
    anchorYOffset: number;
    lineClamp: number;
    fontSize: number;
  }
> = {
  ae1: {
    width: 196,
    height: 94,
    y: 30,
    xInset: 18,
    anchorXOffset: -72,
    anchorYOffset: 158,
    lineClamp: 3,
    fontSize: 12,
  },
  ae2: {
    width: 196,
    height: 94,
    y: 30,
    xInset: 18,
    anchorXOffset: 72,
    anchorYOffset: 158,
    lineClamp: 3,
    fontSize: 12,
  },
  ae3: {
    width: 260,
    height: 138,
    y: 24,
    xInset: 42,
    anchorXOffset: 0,
    anchorYOffset: 118,
    lineClamp: 5,
    fontSize: 13,
  },
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function ThoughtBubble({ agentId, text, visible, charsPerSecond = 90 }: Props) {
  const config = AGENT_VISUAL[agentId];
  const bubble = BUBBLE_CONFIG[agentId];
  const [displayed, setDisplayed] = useState('');

  useEffect(() => {
    if (!visible || !text) {
      setDisplayed('');
      return;
    }
    setDisplayed('');
    let cursor = 0;
    const intervalMs = 1000 / charsPerSecond;
    const timer = setInterval(() => {
      cursor += 1;
      setDisplayed(text.slice(0, cursor));
      if (cursor >= text.length) clearInterval(timer);
    }, intervalMs);
    return () => clearInterval(timer);
  }, [text, visible, charsPerSecond]);

  const anchor = {
    x: config.x + bubble.anchorXOffset,
    y: config.y - bubble.anchorYOffset,
  };
  const preferredX =
    config.bubbleSide === 'left'
      ? config.x - bubble.width + bubble.xInset
      : config.bubbleSide === 'right'
        ? config.x - bubble.xInset
        : config.x - bubble.width / 2;
  const bubbleX = clamp(
    preferredX,
    SCENE_PADDING,
    SCENE_WIDTH - bubble.width - SCENE_PADDING,
  );
  const bubbleY = bubble.y;
  const tailBaseX = clamp(anchor.x, bubbleX + 30, bubbleX + bubble.width - 30);
  const tailBaseY = bubbleY + bubble.height;

  return (
    <AnimatePresence>
      {visible && (
        <motion.g
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8, y: -10 }}
          transition={{ type: 'spring', stiffness: 280, damping: 22 }}
          style={{ transformOrigin: `${config.x}px ${config.y}px` }}
        >
          {/* Sombra del bocadillo */}
          <rect
            x={bubbleX + 3}
            y={bubbleY + 4}
            width={bubble.width}
            height={bubble.height}
            rx="18"
            fill="rgba(0, 0, 0, 0.28)"
          />

          {/* Cuerpo del bocadillo */}
          <rect
            x={bubbleX}
            y={bubbleY}
            width={bubble.width}
            height={bubble.height}
            rx="18"
            fill="rgba(15, 23, 42, 0.92)"
            stroke={config.accent}
            strokeWidth="2.5"
          />

          {/* Colita triangular apuntando a la cabeza del personaje. */}
          <path
            d={`M ${tailBaseX - 13} ${tailBaseY - 3} L ${tailBaseX + 13} ${tailBaseY - 3} L ${anchor.x} ${anchor.y} Z`}
            fill="rgba(15, 23, 42, 0.92)"
            stroke={config.accent}
            strokeWidth="2.5"
          />

          {/* Texto */}
          <foreignObject
            x={bubbleX + 14}
            y={bubbleY + 12}
            width={bubble.width - 28}
            height={bubble.height - 24}
          >
            <div
              style={{
                fontSize: `${bubble.fontSize}px`,
                lineHeight: 1.45,
                color: '#e2e8f0',
                fontFamily: 'var(--font-sans)',
                overflow: 'hidden',
                display: '-webkit-box',
                WebkitLineClamp: bubble.lineClamp,
                WebkitBoxOrient: 'vertical',
                textOverflow: 'ellipsis',
              }}
            >
              <span
                style={{
                  display: 'block',
                  fontSize: '10px',
                  fontWeight: 700,
                  color: config.accent,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  marginBottom: '4px',
                }}
              >
                {config.label} dice…
              </span>
              {displayed}
              {displayed.length < text.length && (
                <span
                  style={{
                    display: 'inline-block',
                    width: '2px',
                    height: '14px',
                    background: config.accent,
                    marginLeft: '2px',
                    animation: 'blink 0.8s infinite',
                    verticalAlign: 'middle',
                  }}
                />
              )}
            </div>
          </foreignObject>
        </motion.g>
      )}
    </AnimatePresence>
  );
}
