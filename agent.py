# =============================================================================
# agent.py — Agente Q-Learning com política epsilon-greedy
# =============================================================================
import random
import numpy as np
import config


class QLearningAgent:
    """
    Agente Q-Learning tabular com política epsilon-greedy e decaimento de
    epsilon ao longo dos episódios.
    """

    def __init__(self, num_states, num_actions):
        self.num_states  = num_states
        self.num_actions = num_actions

        # Hiperparâmetros (podem ser alterados em tempo real pela UI)
        self.alpha   = config.ALPHA
        self.gamma   = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.eps_min = config.EPSILON_MIN
        self.eps_dec = config.EPSILON_DECAY

        # Tabela Q: shape (rows, cols, num_actions)
        rows, cols = config.GRID_ROWS, config.GRID_COLS
        self.Q = np.zeros((rows, cols, num_actions), dtype=np.float32)

        # Histórico para métricas
        self.episode_rewards: list[float] = []
        self.episode_steps:   list[int]   = []
        self.episode_count    = 0
        self.success_count    = 0   # episódios que chegaram ao objetivo

    # ------------------------------------------------------------------
    def choose_action(self, state) -> int:
        """Política epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """Regra de atualização Q-Learning clássica."""
        r,  c  = state
        nr, nc = next_state

        current_q  = self.Q[r, c, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.Q[nr, nc]))

        self.Q[r, c, action] += self.alpha * (target - current_q)

    # ------------------------------------------------------------------
    def decay_epsilon(self):
        """Reduz epsilon após cada episódio."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

    # ------------------------------------------------------------------
    def record_episode(self, total_reward, steps, reached_goal):
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.episode_count += 1
        if reached_goal:
            self.success_count += 1

    # ------------------------------------------------------------------
    def recent_avg_reward(self, window=20) -> float:
        if not self.episode_rewards:
            return 0.0
        last = self.episode_rewards[-window:]
        return sum(last) / len(last)

    def success_rate(self, window=20) -> float:
        """Taxa de sucesso nos últimos `window` episódios."""
        if self.episode_count == 0:
            return 0.0
        last = self.episode_rewards[-window:]
        # Considera sucesso quando a recompensa inclui +100
        successes = sum(1 for r in last if r >= config.REWARD_GOAL * 0.5)
        return successes / len(last)

    # ------------------------------------------------------------------
    def best_action(self, state) -> int:
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    def value(self, state) -> float:
        r, c = state
        return float(np.max(self.Q[r, c]))

    # ------------------------------------------------------------------
    def reset(self):
        """Reinicia a tabela Q e todos os contadores."""
        self.Q[:] = 0.0
        self.epsilon = config.EPSILON_START
        self.episode_rewards.clear()
        self.episode_steps.clear()
        self.episode_count  = 0
        self.success_count  = 0
