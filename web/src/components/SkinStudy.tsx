import { useState } from 'react';
const layers = [
  {
    name: 'Epidermis',
    detail: 'Lapisan terluar yang membantu melindungi kulit dan menjaga kelembapan.',
  },
  {
    name: 'Dermis',
    detail: 'Lapisan penopang yang berisi jaringan ikat, pembuluh darah, dan folikel.',
  },
  {
    name: 'Hipodermis',
    detail: 'Jaringan di bawah dermis yang membantu melindungi dan menyangga tubuh.',
  },
];
export default function SkinStudy({ compact = false }: { compact?: boolean }) {
  const [active, setActive] = useState(0);
  return (
    <div className={`skin-study ${compact ? 'compact' : ''}`}>
      <svg
        viewBox="0 0 600 500"
        role="img"
        aria-label="Ilustrasi konseptual tiga lapisan kulit, bukan hasil pemindaian"
      >
        <defs>
          <pattern id="grain" width="19" height="19" patternUnits="userSpaceOnUse">
            <circle cx="4" cy="4" r="0.8" fill="#203b53" opacity=".18" />
            <circle cx="13" cy="12" r="0.6" fill="#203b53" opacity=".13" />
          </pattern>
          <pattern id="cells" width="38" height="28" patternUnits="userSpaceOnUse">
            <path
              d="M0 14 Q10 0 19 14 T38 14 M19 0 Q28 14 38 0 M0 28 Q9 14 19 28"
              fill="none"
              stroke="#846c59"
              strokeWidth=".7"
              opacity=".4"
            />
          </pattern>
          <filter id="softShadow" x="-30%" y="-40%" width="160%" height="180%">
            <feDropShadow dx="0" dy="18" stdDeviation="18" floodColor="#244057" floodOpacity=".1" />
          </filter>
        </defs>
        <g
          className={`skin-layer layer-bottom ${active === 2 ? 'selected' : ''}`}
          filter="url(#softShadow)"
        >
          <path
            d="M69 283 Q112 252 168 242 L487 283 Q544 290 523 319 L398 413 Q374 436 321 423 L82 349 Q43 333 69 309Z"
            fill="#d7c2a8"
          />
          <path
            d="M69 283 Q112 252 168 242 L487 283 Q544 290 523 309 L398 393 Q374 415 321 401 L82 328 Q43 313 69 283Z"
            fill="#e9ddc9"
          />
          <path
            d="M69 283 Q112 252 168 242 L487 283 Q544 290 523 309 L398 393 Q374 415 321 401 L82 328 Q43 313 69 283Z"
            fill="url(#cells)"
          />
        </g>
        <g
          className={`skin-layer layer-middle ${active === 1 ? 'selected' : ''}`}
          filter="url(#softShadow)"
        >
          <path
            d="M69 207 Q112 176 168 166 L487 207 Q544 214 523 243 L398 337 Q374 360 321 347 L82 273 Q43 257 69 233Z"
            fill="#b88c79"
          />
          <path
            d="M69 207 Q112 176 168 166 L487 207 Q544 214 523 233 L398 317 Q374 339 321 325 L82 252 Q43 237 69 207Z"
            fill="#dcb4a0"
          />
          <path
            d="M117 221 Q197 185 236 234 T364 267 T473 231 M103 235 Q187 201 219 249 T355 281 T488 242 M153 207 Q204 246 188 272 M288 200 Q255 253 315 300 M411 217 Q364 274 392 296"
            stroke="#aa776b"
            strokeWidth="1.4"
            fill="none"
            opacity=".55"
          />
        </g>
        <g
          className={`skin-layer layer-top ${active === 0 ? 'selected' : ''}`}
          filter="url(#softShadow)"
        >
          <path
            d="M69 131 Q112 100 168 90 L487 131 Q544 138 523 167 L398 261 Q374 284 321 271 L82 197 Q43 181 69 157Z"
            fill="#ba9c84"
          />
          <path
            d="M69 131 Q112 100 168 90 L487 131 Q544 138 523 157 L398 241 Q374 263 321 249 L82 176 Q43 161 69 131Z"
            fill="#dfc4aa"
          />
          <path
            d="M69 131 Q112 100 168 90 L487 131 Q544 138 523 157 L398 241 Q374 263 321 249 L82 176 Q43 161 69 131Z"
            fill="url(#grain)"
          />
          {[0, 1, 2, 3, 4, 5, 6].map((i) => (
            <path
              key={i}
              d={`M${120 + i * 45} ${126 + i * 5}q-15 18 8 30t-4 28`}
              fill="none"
              stroke="#b28e75"
              strokeWidth=".9"
              opacity=".7"
            />
          ))}
        </g>
        <path
          d="M456 112h58v-40M496 224h46M454 354h65v45"
          fill="none"
          stroke="#627988"
          strokeWidth=".8"
        />
        <circle cx="456" cy="112" r="3" fill="#203b53" />
        <circle cx="496" cy="224" r="3" fill="#203b53" />
        <circle cx="454" cy="354" r="3" fill="#203b53" />
        <text x="519" y="65" className="diagram-number">
          01
        </text>
        <text x="548" y="228" className="diagram-number">
          02
        </text>
        <text x="509" y="418" className="diagram-number">
          03
        </text>
      </svg>
      {!compact && (
        <>
          <div className="layer-controls" aria-label="Jelajahi lapisan kulit">
            {layers.map((layer, i) => (
              <button
                key={layer.name}
                type="button"
                aria-pressed={active === i}
                onClick={() => setActive(i)}
              >
                <span>0{i + 1}</span>
                {layer.name}
              </button>
            ))}
          </div>
          <p className="layer-description" aria-live="polite">
            {layers[active].detail}
          </p>
        </>
      )}
    </div>
  );
}
