"""Agent DQN (Deep Q-Network) per al Snake.

La IA no veu la pantalla: rep 20 nombres que descriuen la situació, i una
xarxa neuronal petita estima quina puntuació futura (valor Q) espera obtenir
amb cada una de les 3 accions (recte, dreta, esquerra).

Per saber on té el cos, per a cada moviment possible calcula:
  - quant espai lliure li quedaria (flood fill),
  - si encara podria arribar a la cua (la sortida segura que no es tanca mai),
  - quantes caselles pot avançar en línia recta abans de xocar.
"""
import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game import DIRS, SnakeGame

STATE_SIZE = 20
N_ACTIONS = 3


def _flood(game: SnakeGame, start, body) -> tuple[int, bool]:
    """Compta les caselles accessibles des de `start` i diu si s'arriba a la cua.

    `body` és la serp tal com quedaria després del moviment (body[0] = start).
    La cua (body[-1]) es mourà, així que es pot trepitjar.
    """
    obstacles = set(body[1:-1])
    tail = body[-1]
    seen = {start}
    queue = deque([start])
    reaches_tail = False
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if nxt == tail and len(body) > 1:
                reaches_tail = True
            if (nxt in seen or nxt in obstacles
                    or not (0 <= nxt[0] < game.w and 0 <= nxt[1] < game.h)):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen), reaches_tail


def _look_ahead(game: SnakeGame, d: int):
    """Espai lliure, accés a la cua i distància a l'obstacle si es mou en direcció d."""
    hx, hy = game.snake[0]
    dx, dy = DIRS[d]
    head = (hx + dx, hy + dy)
    if game.is_collision(head):
        return 0.0, 0.0, 0.0

    eats = head == game.food
    body = [head] + (game.snake if eats else game.snake[:-1])
    free_total = game.w * game.h - len(body) + 2  # caselles buides + cap + cua
    space, tail_ok = _flood(game, head, body)

    dist, (x, y) = 1, head
    while not game.is_collision((x + dx, y + dy)) and dist < max(game.w, game.h):
        x, y, dist = x + dx, y + dy, dist + 1

    return space / free_total, float(tail_ok), dist / max(game.w, game.h)


def get_state(game: SnakeGame) -> np.ndarray:
    hx, hy = game.snake[0]
    d = game.dir
    rel = [d, (d + 1) % 4, (d - 1) % 4]  # recte, dreta, esquerra

    def danger(di):
        v = DIRS[di]
        return game.is_collision((hx + v[0], hy + v[1]))

    fx, fy = game.food if game.food else (hx, hy)
    look = [_look_ahead(game, di) for di in rel]

    state = [
        # Perill immediat (recte, dreta, esquerra)
        *(danger(di) for di in rel),
        # Direcció actual
        d == 0, d == 1, d == 2, d == 3,  # dreta, avall, esquerra, amunt
        # On és el menjar respecte del cap
        fx > hx, fy > hy, fx < hx, fy < hy,  # dreta, sota, esquerra, dalt
        # Coneixement del propi cos (recte, dreta, esquerra)
        *(l[0] for l in look),  # espai lliure (0-1)
        *(l[1] for l in look),  # pot arribar a la cua?
        *(l[2] for l in look),  # distància a l'obstacle (0-1)
    ]
    return np.array(state, dtype=np.float32)


class QNet(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.95, memory: int = 100_000,
                 batch_size: int = 512, target_sync: int = 1000):
        self.model = QNet()
        # Xarxa objectiu: una còpia que s'actualitza de tant en tant perquè
        # l'objectiu d'aprenentatge no es mogui a cada pas (més estable).
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.updates = 0
        self.memory = deque(maxlen=memory)
        self.epsilon = 1.0  # probabilitat de fer una acció a l'atzar (exploració)

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            q = self.model(torch.from_numpy(state).unsqueeze(0))
        return int(q.argmax().item())

    def ranked_actions(self, state: np.ndarray) -> list[int]:
        """Accions ordenades de la que la IA creu millor a la pitjor."""
        with torch.no_grad():
            q = self.model(torch.from_numpy(state).unsqueeze(0))[0]
        return [int(a) for a in q.argsort(descending=True)]

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self, batch):
        s, a, r, s2, done = zip(*batch)
        s = torch.from_numpy(np.array(s))
        s2 = torch.from_numpy(np.array(s2))
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: la xarxa principal tria l'acció, la xarxa objectiu la valora.
            best = self.model(s2).argmax(1, keepdim=True)
            future = self.target(s2).gather(1, best).squeeze(1)
            # Equació de Bellman: recompensa ara + valor descomptat del millor futur
            target = r + self.gamma * future * (1 - done)

        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.model.state_dict())
        return loss.item()

    def train_replay(self):
        """Aprèn d'un lot d'experiències antigues triades a l'atzar (experience replay)."""
        if len(self.memory) < self.batch_size:
            return 0.0
        return self.train_step(random.sample(self.memory, self.batch_size))

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self.epsilon = 0.0
