"""Proves del joc i de l'estat. Executa: python -m pytest  (o python test_game.py)"""
from agent import get_state
import random

from game import REWARD_DEATH, REWARD_FOOD, SnakeGame
from guardian import Guardian, hamiltonian_cycle


def test_reset():
    g = SnakeGame(10, 10, seed=0)
    assert len(g.snake) == 3 and g.score == 0 and g.food not in g.snake


def test_eat_food():
    g = SnakeGame(10, 10, seed=0)
    hx, hy = g.snake[0]
    g.food = (hx + 1, hy)
    r = g.step(0)
    assert r.reward == REWARD_FOOD and g.score == 1 and len(g.snake) == 4


def test_wall_death():
    g = SnakeGame(10, 10, seed=0)
    g.food = (0, 0)
    for _ in range(10):
        r = g.step(0)
        if r.done:
            break
    assert r.done and r.reward == REWARD_DEATH


def test_turns():
    g = SnakeGame(10, 10, seed=0)
    g.step(1)
    assert g.dir == 1  # dreta -> avall
    g.step(2)
    assert g.dir == 0  # avall -> dreta


def test_state():
    g = SnakeGame(10, 10, seed=0)
    hx, hy = g.snake[0]
    g.food = (hx + 2, hy)
    s = get_state(g)
    assert s.shape == (20,)
    assert list(s[3:7]) == [1, 0, 0, 0]  # va cap a la dreta
    assert s[7] == 1 and s[9] == 0  # menjar a la dreta


def test_detects_trap():
    """Si va recte entra a la cantonada (0,0), una bossa tancada pel seu propi cos."""
    g = SnakeGame(8, 8, seed=0)
    g.snake = [(1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    g.dir = 2  # cap a l'esquerra
    g.food = (7, 7)
    s = get_state(g)
    assert s[0] == 0  # recte no és un xoc immediat...
    assert s[11] < 0.05 and s[14] == 0  # ...però és una trampa: 1 casella i sense camí a la cua


def test_open_space():
    g = SnakeGame(10, 10, seed=0)
    g.food = (0, 0)
    s = get_state(g)
    assert s[11] > 0.9 and s[14] == 1  # recte: gairebé tot el tauler lliure i cua accessible


def test_state_values_in_range():
    g = SnakeGame(10, 10, seed=3)
    for _ in range(30):
        s = get_state(g)
        assert ((s >= 0) & (s <= 1)).all()
        if g.step(g.rng.randrange(3)).done:
            g.reset()


def test_hamiltonian_cycle():
    for w, h in [(20, 20), (6, 4), (5, 6), (6, 5)]:
        c = hamiltonian_cycle(w, h)
        assert len(c) == len(set(c)) == w * h
        for a, b in zip(c, c[1:] + c[:1]):  # cada pas és a una casella veïna
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_guardian_always_wins():
    """Fins i tot si la 'IA' tria a l'atzar, el guardià garanteix la victòria."""
    rng = random.Random(0)
    for seed in range(30):
        g = SnakeGame(8, 8, seed=seed)
        guard = Guardian(g)
        while True:
            prefs = [0, 1, 2]
            rng.shuffle(prefs)
            if g.step(guard.choose(prefs)).done:
                break
        assert g.food is None and g.score == 8 * 8 - 3, f"llavor {seed}: {g.score} punts"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
