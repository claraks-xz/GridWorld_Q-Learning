# =============================================================================
# config.py — Hiperparâmetros e configuração do mapa
# =============================================================================

# ---------------------  Dimensões da grade  -----------------------------------
GRID_ROWS = 8
GRID_COLS = 8

# ---------------------  Hiperparâmetros Q-Learning  --------------------------
ALPHA          = 0.1      # Taxa de aprendizado
GAMMA          = 0.95     # Fator de desconto
EPSILON_START  = 1.0      # Exploração inicial (100 %)
EPSILON_MIN    = 0.05     # Exploração mínima
EPSILON_DECAY  = 0.995    # Multiplicador por episódio
NUM_EPISODES   = 2000     # Total de episódios de treinamento
MAX_STEPS      = 200      # Passos máximos por episódio

# ---------------------  Recompensas  -----------------------------------------
REWARD_GOAL    = +100     # Chegar ao objetivo
REWARD_TRAP    = -50      # Cair numa armadilha
REWARD_WALL    = -5       # Tentar andar na parede (rebate)
REWARD_STEP    = -1       # Custo por passo (incentiva caminhos curtos)

# ---------------------  Estocásticidade  -------------------------------------
# Se STOCHASTIC = True, 20 % das vezes o agente vai em direção aleatória
STOCHASTIC     = False
STOCH_PROB     = 0.20     # Probabilidade de ação aleatória no ambiente

# ---------------------  Mapa padrão (8 x 8)  ---------------------------------
# Legenda das células:
#   'S' = Start (início do agente)
#   'G' = Goal  (objetivo)
#   'W' = Wall  (parede / obstáculo)
#   'T' = Trap  (armadilha / perigo)
#   '.' = Espaço livre

DEFAULT_MAP = [
    ['S', '.', '.', 'W', '.', '.', '.', '.'],
    ['.', 'W', '.', 'W', '.', 'W', 'W', '.'],
    ['.', 'W', '.', '.', '.', '.', 'W', '.'],
    ['.', '.', '.', 'W', 'W', '.', '.', '.'],
    ['W', 'T', '.', '.', '.', '.', 'T', 'W'],
    ['.', '.', 'W', 'W', '.', 'W', '.', '.'],
    ['.', 'T', '.', '.', '.', '.', 'W', '.'],
    ['.', '.', '.', 'W', '.', '.', '.', 'G'],
]

# ---------------------  Interface visual  ------------------------------------
CELL_SIZE      = 72       # Pixels por célula
PANEL_WIDTH    = 310      # Largura do painel lateral (px)

# Velocidades: passos executados antes de cada redesenho
SPEED_MODES    = [1, 10, 100]  # 1=normal, 10=rápido, 100=turbo

# Delay (ms) entre frames para cada velocidade
SPEED_DELAYS   = [60, 20, 1]

# ---------------------  Cores (hex strings para Tkinter)  -------------------
C_BG        = "#1E1E2E"
C_GRID      = "#3C3C50"
C_FREE      = "#2D2D41"
C_WALL      = "#5A5A78"
C_GOAL      = "#50C878"
C_TRAP      = "#DC4646"
C_START     = "#6490DC"
C_AGENT     = "#FFDC32"
C_PANEL_BG  = "#14142A"
C_TEXT      = "#DCDCF0"
C_ACCENT    = "#82AAF0"
C_ARROW     = "#FFC832"
C_HEAT_LOW  = "#28285A"
C_HEAT_HIGH = "#FF6428"
C_BTN       = "#3C3C64"
C_BTN_HOV   = "#5050A0"
C_SUCCESS   = "#64DC64"
C_WARN      = "#FFB450"
