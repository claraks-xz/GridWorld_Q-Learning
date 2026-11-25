# =============================================================================
# main.py — Ponto de entrada: inicializa ambiente, agente e loop principal
# =============================================================================
"""
GridWorld Q-Learning — Demo Interativo
======================================
Execute:  python main.py

Dependências: apenas Python padrão + numpy
  pip install numpy

Controles:
  Espaço   Pausar / Continuar treinamento
  R        Resetar agente e tabela Q
  V        Alternar visualização (política | mapa de calor | nenhuma)
  ↑/↓      Aumentar / Diminuir velocidade
  E        Ativar/desativar modo de edição da grade
  T        Alternar ferramenta (parede / armadilha)
  Clique   Em modo edição: adicionar/remover parede ou armadilha
"""

import config
from env   import GridWorld
from agent import QLearningAgent
from ui    import GridUI


def main():
    # 1. Cria o ambiente com o mapa padrão
    env = GridWorld()

    # 2. Cria o agente Q-Learning
    agent = QLearningAgent(
        num_states  = env.num_states(),
        num_actions = env.NUM_ACTIONS,
    )

    # 3. Cria e abre a interface visual (Tkinter)
    gui = GridUI(env, agent)
    gui.run()


if __name__ == "__main__":
    main()
