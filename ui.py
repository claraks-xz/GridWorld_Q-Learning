# =============================================================================
# ui.py — Interface Tkinter: renderização, painel lateral e controles
# =============================================================================
"""
Interface 100% Tkinter (sem dependências externas).
Renderiza grid, agente, política/mapa de calor e painel de métricas.
"""

import math
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import config
from env   import GridWorld
from agent import QLearningAgent


# ---------------------------------------------------------------------------
# Utilitários de cor
# ---------------------------------------------------------------------------

def hex2rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def lerp_color(c1_hex: str, c2_hex: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = hex2rgb(c1_hex)
    r2, g2, b2 = hex2rgb(c2_hex)
    return rgb2hex(r1 + (r2 - r1) * t,
                   g1 + (g2 - g1) * t,
                   b1 + (b2 - b1) * t)


# Direção das setas: (dx_canvas, dy_canvas) para cada ação
# Ações: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita
ARROW_DIRS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
ARROW_SYMBOLS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


# ---------------------------------------------------------------------------
class GridUI:
    """Janela principal Tkinter com Canvas para a grade e Frame para o painel."""

    def __init__(self, env: GridWorld, agent: QLearningAgent):
        self.env   = env
        self.agent = agent
        self.cell  = config.CELL_SIZE
        self.rows  = env.rows
        self.cols  = env.cols

        # Estado da UI
        self.paused       = False
        self.view_mode    = "policy"   # "policy" | "heat" | "none"
        self.speed_idx    = 0
        self.edit_mode    = False
        self.edit_tool    = "wall"     # "wall" | "trap"
        self.running      = True

        # Dados do episódio atual
        self.cur_state    = env.reset()
        self.cur_step     = 0
        self.cur_reward   = 0.0
        self.episode_done = False

        # Cache de IDs de itens do canvas por célula
        self._cell_ids: dict = {}

        self._build_window()
        self._build_bindings()
        self._schedule_loop()

    # ------------------------------------------------------------------
    # Construção da janela
    # ------------------------------------------------------------------
    def _build_window(self):
        grid_w = self.cols * self.cell
        grid_h = self.rows * self.cell

        self.root = tk.Tk()
        self.root.title("GridWorld Q-Learning  |  IIA Demo")
        self.root.configure(bg=config.C_PANEL_BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Fontes ---
        self.f_title  = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.f_bold   = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_norm   = tkfont.Font(family="Segoe UI", size=10)
        self.f_small  = tkfont.Font(family="Segoe UI", size=9)
        self.f_arrow  = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.f_emoji  = tkfont.Font(family="Segoe UI Emoji", size=16)

        # --- Canvas da grade ---
        self.canvas = tk.Canvas(
            self.root, width=grid_w, height=grid_h,
            bg=config.C_BG, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # --- Painel lateral ---
        self.panel = tk.Frame(
            self.root, width=config.PANEL_WIDTH, bg=config.C_PANEL_BG
        )
        self.panel.grid(row=0, column=1, sticky="nsew", padx=0)
        self.panel.grid_propagate(False)

        self._build_panel(grid_h)

    def _build_panel(self, height):
        p = self.panel
        pad = dict(padx=10, pady=2, anchor="w")

        # Título
        tk.Label(p, text="GridWorld Q-Learning", font=self.f_title,
                 fg=config.C_ACCENT, bg=config.C_PANEL_BG).pack(pady=(12, 2), padx=10, anchor="w")
        self._sep(p)

        # --- Variáveis de texto reativos ---
        self.var_episode = tk.StringVar(value="Episódio: 0 / 0")
        self.var_step    = tk.StringVar(value="Passo: 0")
        self.var_reward  = tk.StringVar(value="Recompensa: 0.0")
        self.var_epsilon = tk.StringVar(value="ε: 1.0000")
        self.var_alpha   = tk.StringVar(value=f"α: {config.ALPHA}")
        self.var_gamma   = tk.StringVar(value=f"γ: {config.GAMMA}")
        self.var_avg     = tk.StringVar(value="Média (20ep): 0.0")
        self.var_succ    = tk.StringVar(value="Sucesso (20ep): 0%")
        self.var_status  = tk.StringVar(value="▶ TREINANDO")
        self.var_view    = tk.StringVar(value="Visão: Política (setas)")
        self.var_speed   = tk.StringVar(value="Velocidade: 1×")
        self.var_edit    = tk.StringVar(value="")

        # Bloco episódio
        tk.Label(p, textvariable=self.var_episode, font=self.f_bold,
                 fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)

        # Barra de progresso (Canvas pequeno)
        self.prog_canvas = tk.Canvas(p, height=10, bg=config.C_GRID,
                                     highlightthickness=0)
        self.prog_canvas.pack(padx=10, pady=2, fill="x")
        self.prog_bar = self.prog_canvas.create_rectangle(0, 0, 0, 10,
                                                           fill=config.C_ACCENT, width=0)

        tk.Label(p, textvariable=self.var_step,   font=self.f_norm,
                 fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)
        tk.Label(p, textvariable=self.var_reward, font=self.f_norm,
                 fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)
        self._sep(p)

        # Hiperparâmetros
        tk.Label(p, text="Hiperparâmetros", font=self.f_bold,
                 fg=config.C_ACCENT, bg=config.C_PANEL_BG).pack(**pad)
        for var in (self.var_epsilon, self.var_alpha, self.var_gamma):
            tk.Label(p, textvariable=var, font=self.f_norm,
                     fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)
        self._sep(p)

        # Métricas
        tk.Label(p, text="Métricas", font=self.f_bold,
                 fg=config.C_ACCENT, bg=config.C_PANEL_BG).pack(**pad)
        for var in (self.var_avg, self.var_succ):
            tk.Label(p, textvariable=var, font=self.f_norm,
                     fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)
        self._sep(p)

        # Status
        self.lbl_status = tk.Label(p, textvariable=self.var_status, font=self.f_bold,
                                   fg=config.C_SUCCESS, bg=config.C_PANEL_BG)
        self.lbl_status.pack(**pad)
        tk.Label(p, textvariable=self.var_view,  font=self.f_norm,
                 fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)
        tk.Label(p, textvariable=self.var_speed, font=self.f_norm,
                 fg=config.C_TEXT, bg=config.C_PANEL_BG).pack(**pad)
        self.lbl_edit = tk.Label(p, textvariable=self.var_edit, font=self.f_bold,
                                  fg=config.C_WARN, bg=config.C_PANEL_BG)
        self.lbl_edit.pack(**pad)
        self._sep(p)

        # Legenda
        tk.Label(p, text="Legenda", font=self.f_bold,
                 fg=config.C_ACCENT, bg=config.C_PANEL_BG).pack(**pad)
        leg = [("🏆", "Objetivo  (+100)", config.C_GOAL),
               ("💀", "Armadilha (−50)",  config.C_TRAP),
               ("⬛", "Parede",           config.C_WALL),
               ("🤖", "Agente",           config.C_AGENT)]
        for icon, lbl, col in leg:
            fr = tk.Frame(p, bg=config.C_PANEL_BG)
            fr.pack(padx=10, pady=1, anchor="w")
            tk.Label(fr, text=icon, font=self.f_emoji, fg=col,
                     bg=config.C_PANEL_BG).pack(side="left")
            tk.Label(fr, text=lbl,  font=self.f_small, fg=config.C_TEXT,
                     bg=config.C_PANEL_BG).pack(side="left", padx=4)

        self._sep(p)

        # Atalhos
        tk.Label(p, text="Atalhos de teclado", font=self.f_bold,
                 fg=config.C_ACCENT, bg=config.C_PANEL_BG).pack(**pad)
        shortcuts = [
            ("Espaço", "Pausar / Continuar"),
            ("R",      "Resetar agente"),
            ("V",      "Alternar visão"),
            ("↑/↓",    "Velocidade"),
            ("E",      "Modo edição"),
            ("T",      "Trocar ferramenta"),
        ]
        for key, desc in shortcuts:
            fr = tk.Frame(p, bg=config.C_PANEL_BG)
            fr.pack(padx=10, pady=0, anchor="w")
            tk.Label(fr, text=f"[{key}]", font=self.f_small, fg=config.C_ACCENT,
                     bg=config.C_PANEL_BG, width=8, anchor="w").pack(side="left")
            tk.Label(fr, text=desc, font=self.f_small, fg=config.C_TEXT,
                     bg=config.C_PANEL_BG).pack(side="left")

        self._sep(p)

        # Botões
        self._add_btn(p, "⏸  Pausar / Continuar  [Espaço]", self._toggle_pause)
        self._add_btn(p, "🔄  Resetar agente  [R]",          self._do_reset,   config.C_TRAP)
        self._add_btn(p, "👁  Alternar visão  [V]",           self._cycle_view)
        self._add_btn(p, "⚡  Aumentar velocidade  [↑]",      self._inc_speed)
        self._add_btn(p, "🐢  Reduzir velocidade  [↓]",       self._dec_speed)
        self._add_btn(p, "✏  Modo Edição  [E]",               self._toggle_edit, config.C_SUCCESS)
        self._add_btn(p, "🧱  Trocar ferramenta  [T]",         self._toggle_tool, "#806030")

    def _sep(self, parent):
        tk.Frame(parent, height=1, bg=config.C_GRID).pack(fill="x", padx=8, pady=4)

    def _add_btn(self, parent, label, cmd, bg=None):
        bg = bg or config.C_BTN
        btn = tk.Button(
            parent, text=label, command=cmd,
            font=self.f_small, fg=config.C_TEXT, bg=bg,
            activebackground=config.C_BTN_HOV, activeforeground=config.C_TEXT,
            relief="flat", cursor="hand2", pady=4
        )
        btn.pack(fill="x", padx=10, pady=2)

    # ------------------------------------------------------------------
    # Bindings de teclado
    # ------------------------------------------------------------------
    def _build_bindings(self):
        self.root.bind("<space>",   lambda e: self._toggle_pause())
        self.root.bind("<r>",       lambda e: self._do_reset())
        self.root.bind("<R>",       lambda e: self._do_reset())
        self.root.bind("<v>",       lambda e: self._cycle_view())
        self.root.bind("<V>",       lambda e: self._cycle_view())
        self.root.bind("<e>",       lambda e: self._toggle_edit())
        self.root.bind("<E>",       lambda e: self._toggle_edit())
        self.root.bind("<t>",       lambda e: self._toggle_tool())
        self.root.bind("<T>",       lambda e: self._toggle_tool())
        self.root.bind("<Up>",      lambda e: self._inc_speed())
        self.root.bind("<Down>",    lambda e: self._dec_speed())
        self.root.focus_set()

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    def _toggle_pause(self):
        self.paused = not self.paused

    def _do_reset(self):
        self.agent.reset()
        self.cur_state    = self.env.reset()
        self.cur_step     = 0
        self.cur_reward   = 0.0
        self.episode_done = False
        self.paused       = False

    def _cycle_view(self):
        modes = ["none", "policy", "heat"]
        self.view_mode = modes[(modes.index(self.view_mode) + 1) % len(modes)]

    def _inc_speed(self):
        self.speed_idx = min(self.speed_idx + 1, len(config.SPEED_MODES) - 1)

    def _dec_speed(self):
        self.speed_idx = max(self.speed_idx - 1, 0)

    def _toggle_edit(self):
        self.edit_mode = not self.edit_mode

    def _toggle_tool(self):
        self.edit_tool = "trap" if self.edit_tool == "wall" else "wall"

    def _on_canvas_click(self, event):
        if not self.edit_mode:
            return
        c = event.x // self.cell
        r = event.y // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.edit_tool == "wall":
                self.env.toggle_wall(r, c)
            else:
                self.env.toggle_trap(r, c)
            self.agent.reset()
            self.cur_state    = self.env.reset()
            self.cur_step     = 0
            self.cur_reward   = 0.0
            self.episode_done = False

    def _on_close(self):
        self.running = False
        self.root.destroy()

    # ------------------------------------------------------------------
    # Loop de treinamento
    # ------------------------------------------------------------------
    def step_training(self):
        """Executa passos de treinamento (respeita pausa e velocidade)."""
        if self.paused:
            return

        speed = config.SPEED_MODES[self.speed_idx]

        for _ in range(speed):
            if self.episode_done:
                reached = (self.env.agent_pos == self.env.goal_pos)
                self.agent.record_episode(self.cur_reward, self.cur_step, reached)
                self.agent.decay_epsilon()

                if self.agent.episode_count >= config.NUM_EPISODES:
                    self.paused = True
                    return

                self.cur_state    = self.env.reset()
                self.cur_step     = 0
                self.cur_reward   = 0.0
                self.episode_done = False

            action = self.agent.choose_action(self.cur_state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.update(self.cur_state, action, reward, next_state, done)

            self.cur_state    = next_state
            self.cur_step    += 1
            self.cur_reward  += reward
            self.episode_done = done

    # ------------------------------------------------------------------
    # Renderização
    # ------------------------------------------------------------------
    def _schedule_loop(self):
        """Loop principal via after() do Tkinter."""
        if not self.running:
            return
        self.step_training()
        self._draw()
        delay = config.SPEED_DELAYS[self.speed_idx]
        self.root.after(delay, self._schedule_loop)

    def _draw(self):
        self._draw_grid()
        self._update_panel()

    # --- Grade --------------------------------------------------------
    def _heat_color(self, r, c):
        """Cor do mapa de calor para célula (r, c)."""
        all_vals = []
        for rr in range(self.rows):
            for cc in range(self.cols):
                if self.env.cell_type(rr, cc) != 'wall':
                    all_vals.append(float(np.max(self.agent.Q[rr, cc])))
        if not all_vals:
            return config.C_HEAT_LOW
        mn, mx = min(all_vals), max(all_vals)
        if mx == mn:
            return config.C_HEAT_LOW
        v = float(np.max(self.agent.Q[r, c]))
        t = (v - mn) / (mx - mn)
        return lerp_color(config.C_HEAT_LOW, config.C_HEAT_HIGH, t)

    def _draw_grid(self):
        canvas = self.canvas
        canvas.delete("all")
        cell = self.cell
        env  = self.env

        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = c * cell, r * cell
                x1, y1 = x0 + cell, y0 + cell
                ct = env.cell_type(r, c)

                # Cor base
                if ct == 'wall':
                    bg = config.C_WALL
                elif ct == 'goal':
                    bg = config.C_GOAL
                elif ct == 'trap':
                    bg = config.C_TRAP
                elif ct == 'start':
                    bg = config.C_START
                elif self.view_mode == "heat":
                    bg = self._heat_color(r, c)
                else:
                    bg = config.C_FREE

                # Retângulo da célula
                canvas.create_rectangle(x0, y0, x1, y1,
                                        fill=bg, outline=config.C_GRID, width=1)

                # Ícones
                cx, cy = x0 + cell // 2, y0 + cell // 2
                if ct == 'goal':
                    canvas.create_text(cx, cy, text="🏆", font=self.f_emoji)
                elif ct == 'trap':
                    canvas.create_text(cx, cy, text="💀", font=self.f_emoji)
                elif ct == 'start' and env.agent_pos != env.start_pos:
                    canvas.create_text(cx, cy, text="S", font=self.f_small,
                                       fill="#C8E6FF")

                # Sobreposição: política (setas) ou mapa de calor (valores)
                if ct not in ('wall', 'goal', 'trap'):
                    if self.view_mode == "policy":
                        self._draw_arrow(canvas, r, c, x0, y0, cell)
                    elif self.view_mode == "heat":
                        v = float(np.max(self.agent.Q[r, c]))
                        canvas.create_text(x1 - 4, y1 - 4,
                                           text=f"{v:.0f}",
                                           font=self.f_small, fill="#E0E0E0",
                                           anchor="se")

        # Agente
        ar, ac = env.agent_pos
        ax = ac * cell + cell // 2
        ay = ar * cell + cell // 2
        r2 = cell // 2 - 6
        canvas.create_oval(ax - r2, ay - r2, ax + r2, ay + r2,
                           fill=config.C_AGENT, outline="#C8A014", width=2)
        canvas.create_text(ax, ay, text="🤖", font=self.f_emoji)

    def _draw_arrow(self, canvas, r, c, x0, y0, cell):
        """Desenha seta da melhor ação, com opacidade proporcional à confiança."""
        best = self.agent.best_action((r, c))
        sym  = ARROW_SYMBOLS[best]
        dx, dy = ARROW_DIRS[best]
        cx, cy = x0 + cell // 2, y0 + cell // 2
        length = cell // 3

        # Confiança: diferença entre melhor e pior Q
        q_row  = self.agent.Q[r, c]
        spread = float(np.max(q_row) - np.min(q_row))

        # Cor mais viva conforme confiança aumenta
        alpha  = min(1.0, spread / 20.0)
        col    = lerp_color(config.C_FREE, config.C_ARROW, alpha)

        # Linha do corpo
        ex = cx + dx * length
        ey = cy + dy * length
        canvas.create_line(cx, cy, ex, ey, fill=col, width=2, arrow="last",
                           arrowshape=(10, 12, 4))

        # Símbolo (redundante mas legível)
        canvas.create_text(cx, cy, text=sym, font=self.f_arrow, fill=col)

    # --- Painel -------------------------------------------------------
    def _update_panel(self):
        ep  = self.agent.episode_count
        tot = config.NUM_EPISODES
        pct = ep / tot if tot > 0 else 0

        self.var_episode.set(f"Episódio: {ep} / {tot}  ({pct*100:.0f}%)")
        self.var_step.set(   f"Passo atual: {self.cur_step}")
        self.var_reward.set( f"Recompensa: {self.cur_reward:.1f}")
        self.var_epsilon.set(f"ε (epsilon): {self.agent.epsilon:.4f}")
        self.var_alpha.set(  f"α (alpha): {self.agent.alpha}")
        self.var_gamma.set(  f"γ (gamma): {self.agent.gamma}")
        self.var_avg.set(    f"Média (20ep): {self.agent.recent_avg_reward(20):.1f}")
        self.var_succ.set(   f"Sucesso (20ep): {self.agent.success_rate(20)*100:.0f}%")

        # Barra de progresso
        bw = self.prog_canvas.winfo_width()
        self.prog_canvas.coords(self.prog_bar, 0, 0, int(bw * pct), 10)

        # Status
        if self.paused:
            self.var_status.set("⏸  PAUSADO")
            self.lbl_status.config(fg=config.C_WARN)
        elif self.agent.episode_count >= config.NUM_EPISODES:
            self.var_status.set("✅  TREINAMENTO CONCLUÍDO")
            self.lbl_status.config(fg=config.C_SUCCESS)
        else:
            self.var_status.set("▶  TREINANDO")
            self.lbl_status.config(fg=config.C_SUCCESS)

        speeds = {0: "1×  🐢", 1: "10×  🐇", 2: "100×  ⚡"}
        self.var_speed.set(f"Velocidade: {speeds.get(self.speed_idx, '?')}")

        views = {"policy": "Política (setas)", "heat": "Mapa de calor", "none": "Sem sobreposição"}
        self.var_view.set(f"Visão: {views[self.view_mode]}")

        if self.edit_mode:
            tool = "Parede" if self.edit_tool == "wall" else "Armadilha"
            self.var_edit.set(f"✏ EDIÇÃO ATIVA — {tool.upper()}")
        else:
            self.var_edit.set("")

    # ------------------------------------------------------------------
    def run(self):
        """Inicia o loop Tkinter."""
        self.root.mainloop()
