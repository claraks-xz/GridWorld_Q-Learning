# =============================================================================
# env.py — Ambiente GridWorld
# =============================================================================
import random
import config


class GridWorld:
    """
    Ambiente GridWorld determinístico (ou estocástico) com suporte a
    paredes, armadilhas e célula objetivo.
    """

    # Ações: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ["Cima", "Baixo", "Esquerda", "Direita"]
    NUM_ACTIONS = 4

    def __init__(self, grid_map=None):
        self.rows = config.GRID_ROWS
        self.cols = config.GRID_COLS
        self.stochastic = config.STOCHASTIC

        # Inicializa o mapa (pode ser editado em tempo real)
        raw = grid_map if grid_map else config.DEFAULT_MAP
        self._init_grid(raw)

        self.agent_pos = self.start_pos
        self.done = False
        self.total_reward = 0.0
        self.steps = 0

    # ------------------------------------------------------------------
    def _init_grid(self, raw_map):
        """Converte o mapa textual em estruturas internas."""
        self.grid = []          # tipo de cada célula: 'free', 'wall', etc.
        self.start_pos = (0, 0)
        self.goal_pos  = (0, 0)
        self.traps     = set()
        self.walls     = set()

        for r, row in enumerate(raw_map):
            grid_row = []
            for c, cell in enumerate(row):
                if cell == 'S':
                    self.start_pos = (r, c)
                    grid_row.append('free')
                elif cell == 'G':
                    self.goal_pos = (r, c)
                    grid_row.append('goal')
                elif cell == 'W':
                    self.walls.add((r, c))
                    grid_row.append('wall')
                elif cell == 'T':
                    self.traps.add((r, c))
                    grid_row.append('trap')
                else:
                    grid_row.append('free')
            self.grid.append(grid_row)

    # ------------------------------------------------------------------
    def reset(self):
        """Reseta o agente para a posição inicial e retorna o estado."""
        self.agent_pos   = self.start_pos
        self.done        = False
        self.total_reward = 0.0
        self.steps       = 0
        return self._state()

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Executa uma ação no ambiente.
        Retorna: (next_state, reward, done, info)
        """
        if self.done:
            return self._state(), 0.0, True, {}

        # Estocásticidade opcional
        if self.stochastic and random.random() < config.STOCH_PROB:
            action = random.randint(0, self.NUM_ACTIONS - 1)

        dr, dc = self.ACTIONS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Verifica limites e paredes
        if not self._in_bounds(nr, nc) or (nr, nc) in self.walls:
            # Rebate: permanece na mesma posição
            reward = config.REWARD_WALL
            next_pos = self.agent_pos
        else:
            next_pos = (nr, nc)
            if next_pos == self.goal_pos:
                reward = config.REWARD_GOAL
                self.done = True
            elif next_pos in self.traps:
                reward = config.REWARD_TRAP
                self.done = True          # armadilha encerra episódio
            else:
                reward = config.REWARD_STEP

        self.agent_pos = next_pos
        self.total_reward += reward
        self.steps += 1

        if self.steps >= config.MAX_STEPS:
            self.done = True

        return self._state(), reward, self.done, {}

    # ------------------------------------------------------------------
    def _state(self):
        return self.agent_pos   # estado = (row, col)

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def num_states(self):
        return self.rows * self.cols

    # ------------------------------------------------------------------
    # Utilitários para edição ao vivo pelo mouse
    # ------------------------------------------------------------------
    def toggle_wall(self, r, c):
        """Alterna parede em (r, c). Protege início e objetivo."""
        if (r, c) in (self.start_pos, self.goal_pos):
            return
        if (r, c) in self.traps:
            self.traps.discard((r, c))
        if (r, c) in self.walls:
            self.walls.discard((r, c))
            self.grid[r][c] = 'free'
        else:
            self.walls.add((r, c))
            self.grid[r][c] = 'wall'

    def toggle_trap(self, r, c):
        """Alterna armadilha em (r, c). Protege início e objetivo."""
        if (r, c) in (self.start_pos, self.goal_pos):
            return
        if (r, c) in self.walls:
            self.walls.discard((r, c))
        if (r, c) in self.traps:
            self.traps.discard((r, c))
            self.grid[r][c] = 'free'
        else:
            self.traps.add((r, c))
            self.grid[r][c] = 'trap'

    def cell_type(self, r, c):
        if (r, c) == self.goal_pos:
            return 'goal'
        if (r, c) == self.start_pos:
            return 'start'
        if (r, c) in self.walls:
            return 'wall'
        if (r, c) in self.traps:
            return 'trap'
        return 'free'
