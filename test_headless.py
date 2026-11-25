# test_headless.py — Valida env.py e agent.py sem Pygame
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch config para valores pequenos
import config
config.NUM_EPISODES = 200
config.MAX_STEPS    = 100

from env   import GridWorld
from agent import QLearningAgent

def run_test():
    env   = GridWorld()
    agent = QLearningAgent(env.num_states(), env.NUM_ACTIONS)

    for ep in range(config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0.0
        for step in range(config.MAX_STEPS):
            action     = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward
            if done:
                break

        reached = (env.agent_pos == env.goal_pos)
        agent.record_episode(total_reward, step + 1, reached)
        agent.decay_epsilon()

    print(f"Episódios treinados: {agent.episode_count}")
    print(f"Epsilon final:       {agent.epsilon:.4f}")
    print(f"Recomp. média (20):  {agent.recent_avg_reward(20):.2f}")
    print(f"Taxa de sucesso(20): {agent.success_rate(20)*100:.0f}%")

    # Testa política final
    state = env.reset()
    path  = [state]
    for _ in range(50):
        action     = agent.best_action(state)
        next_state, reward, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    success = (env.agent_pos == env.goal_pos)
    print(f"Caminho aprendido:   {len(path)} passos  |  Chegou ao objetivo: {success}")
    print("✅ Teste headless passou com sucesso!")
    return True

if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
