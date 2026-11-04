# Left Sidebar Navigation Plan — Concrete Strength AI Lab

## Goal
Move the 4 navigation tabs (**1. Dataset & Scaling**, **2. PCA Simulator**, **3. Regression Comparison**, **4. Strength Predictor**) from the top horizontal bar into a clean, fixed **left sidebar** so they never clip or require horizontal scrolling.

---

## Layout Update

```
┌──────────────────────────┬───────────────────────────────────────────────┐
│ Concrete Strength AI Lab │  Active Section Header                        │
│ PCA & Regression Lab     ├───────────────────────────────────────────────┤
├──────────────────────────┤                                               │
│                          │                                               │
│ ● 1. Dataset & Scaling   │                                               │
│                          │         Main Interactive Content              │
│ ● 2. PCA Simulator       │        (Spacious, Full-Width Cards)           │
│                          │                                               │
│ ● 3. Regression Compare  │                                               │
│                          │                                               │
│ ● 4. Strength Predictor  │                                               │
│                          │                                               │
├──────────────────────────┤                                               │
│ Active PCA: k = 6 / 8    │                                               │
│ Variance: 97.31%         │                                               │
└──────────────────────────┴───────────────────────────────────────────────┘
```

- **Clean Left Sidebar**: Displays the app title, subtitle, the 4 vertically stacked navigation tabs with icons, and a compact PCA status card at the bottom.
- **No Horizontal Scrollbar**: Eliminates the clipped horizontal tab pill container shown in your screenshot.
- **Mobile Friendly**: Includes a clean top menu toggle for smaller screens while keeping the fixed left sidebar on desktop.
