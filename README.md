# GridWorld Q-Learning — Demo Interativo

> Demonstração ao vivo da disciplina de **Inteligência Artificial Aplicada (IIA)**.  
> Interface 100% Tkinter — **sem dependências externas além de `numpy`**.

---

## 📁 Estrutura do Projeto

```
gridworld_qlearning/
├── main.py          # Ponto de entrada
├── env.py           # Ambiente GridWorld (estados, ações, recompensas)
├── agent.py         # Agente Q-Learning (tabela Q, política ε-greedy)
├── ui.py            # Interface Tkinter (grade, painel, controles)
├── config.py        # Hiperparâmetros e configuração do mapa
├── test_headless.py # Teste de lógica sem interface gráfica
└── README.md        # Este arquivo
```

---

## ⚙️ Instalação

```bash
# Apenas uma dependência externa:
pip install numpy

# Tkinter já vem com Python — sem instalação adicional.
```

---

## ▶️ Como Rodar

```bash
cd gridworld_qlearning
python main.py
```

---

## 🎮 Controles Interativos

| Tecla / Ação | Efeito |
|---|---|
| `Espaço` | Pausar / Continuar o treinamento |
| `R` | Resetar agente e tabela Q do zero |
| `V` | Alternar visão: Política (setas) → Mapa de calor → Sem sobreposição |
| `↑` | Aumentar velocidade (1× → 10× → 100× Turbo) |
| `↓` | Reduzir velocidade |
| `E` | Ativar / Desativar modo de edição da grade |
| `T` | Alternar ferramenta de edição (Parede ↔ Armadilha) |
| **Clique na grade** | No modo edição: adiciona ou remove parede / armadilha |
| Botões no painel | Mesmas ações via mouse |

---

## 🗺️ Mapa Padrão (8×8)

```
S . . W . . . .
. W . W . W W .
. W . . . . W .
. . . W W . . .
W T . . . . T W
. . W W . W . .
. T . . . . W .
. . . W . . . G
```

**Legenda:** `S` = Início · `G` = Objetivo (+100) · `W` = Parede · `T` = Armadilha (−50) · `.` = Livre (−1/passo)

---

## 🧠 Algoritmo Q-Learning

A regra de atualização implementada em [`agent.py`](agent.py):

```
Q(s, a) ← Q(s, a) + α × [ r + γ × max_a' Q(s', a') − Q(s, a) ]
```

### Hiperparâmetros (editáveis em [`config.py`](config.py))

| Parâmetro | Valor Padrão | Descrição |
|---|---|---|
| `ALPHA` | `0.1` | Taxa de aprendizado |
| `GAMMA` | `0.95` | Fator de desconto |
| `EPSILON_START` | `1.0` | Exploração inicial (100%) |
| `EPSILON_MIN` | `0.05` | Exploração mínima |
| `EPSILON_DECAY` | `0.995` | Decaimento por episódio |
| `NUM_EPISODES` | `2000` | Total de episódios |
| `MAX_STEPS` | `200` | Passos máximos por episódio |

---

## 💡 Roteiro de Demonstração (20 min)

### 1 — Apresentação do ambiente (~3 min)
- Mostre o mapa: onde é o início (azul), objetivo (verde 🏆), armadilhas (vermelho 💀) e paredes (cinza).
- Explique as 4 ações e as recompensas.

### 2 — Treinamento acelerado (~4 min)
- Pressione `↑` duas vezes para **modo Turbo (100×)**.
- Observe o epsilon caindo e a taxa de sucesso subindo no painel.
- Aguarde ~500 episódios e pressione `↓` para voltar a 1×.

### 3 — Visualizar a política aprendida (~4 min)
- Pressione `V` para ativar **Política (setas)**.
- Mostre como as setas formam um caminho coerente do início ao objetivo, desviando de armadilhas.
- Pressione `V` novamente para o **Mapa de calor** e explique os valores Q por célula.

### 4 — Demonstração ao vivo de adaptação (~5 min)
- Pressione `Espaço` para pausar.
- Pressione `E` para ativar o modo edição.
- **Clique em células** para adicionar uma parede bloqueando o caminho aprendido.
- Pressione `R` para resetar e observe o agente reaprendendo um novo caminho.
- Converta algumas paredes em armadilhas com `T` para mostrar penalidades.

### 5 — Discussão técnica (~4 min)
- Mostre o código de [`agent.py`](agent.py) (linha da atualização Q).
- Explique epsilon-greedy e o trade-off exploração × exploração.
- Comente sobre estocásticidade (`STOCHASTIC = True` em [`config.py`](config.py)).

---

## 🔬 Extensões Fáceis

| O que fazer | Onde mudar |
|---|---|
| Ativar ruído no ambiente | `STOCHASTIC = True` em `config.py` |
| Grade maior (10×10) | `GRID_ROWS`, `GRID_COLS` e `DEFAULT_MAP` em `config.py` |
| Ajustar recompensas | `REWARD_GOAL`, `REWARD_TRAP`, `REWARD_STEP` em `config.py` |
| Mais episódios | `NUM_EPISODES` em `config.py` |
| Epsilon mais lento | `EPSILON_DECAY` próximo de 1.0 (ex.: 0.999) |

---

## 🧪 Teste sem Interface

```bash
python test_headless.py
```

Valida env.py e agent.py sem abrir janela gráfica.
