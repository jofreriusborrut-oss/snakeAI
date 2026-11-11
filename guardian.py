"""Guardià: fa que la serp guanyi SEMPRE (omple tot el tauler).

Idea: un cicle hamiltonià és un camí que passa per totes les caselles del
tauler una sola vegada i torna a l'inici. Si la serp el segueix, mai no xoca.
Però seguir-lo sempre és molt lent, així que la IA pot fer DRECERES, i el
guardià només les accepta si són segures:

  Numerem les caselles en l'ordre del cicle. El cos de la serp sempre està
  entre la cua i el cap (en aquest ordre), i les caselles que hi ha DAVANT del
  cap fins a la cua són buides. Una drecera és segura si:
    1. no avança tant que passi per davant de la cua, i
    2. no salta per sobre del menjar (així sempre hi acaba arribant).

Amb aquestes dues regles es pot demostrar que la serp mai no mor.
"""
from game import DIRS, SnakeGame


def hamiltonian_cycle(w: int, h: int) -> list[tuple[int, int]]:
    """Cicle que recorre tot el tauler: zig-zag per les columnes 1..w-1 i torna per la columna 0."""
    if h % 2 == 1:
        if w % 2 == 1:
            raise ValueError("Cal que l'amplada o l'alçada del tauler sigui parella.")
        return [(x, y) for (y, x) in hamiltonian_cycle(h, w)]  # gira el tauler
    path = []
    for y in range(h):
        xs = range(1, w) if y % 2 == 0 else range(w - 1, 0, -1)
        path += [(x, y) for x in xs]
    path += [(0, y) for y in range(h - 1, -1, -1)]
    return path


class Guardian:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.cycle = hamiltonian_cycle(game.w, game.h)
        self.n = len(self.cycle)
        self.index = {cell: i for i, cell in enumerate(self.cycle)}

    def dist(self, a, b) -> int:
        """Quantes caselles cal avançar pel cicle per anar de a a b."""
        return (self.index[b] - self.index[a]) % self.n

    def safe_moves(self) -> list[int]:
        """Accions (0 recte, 1 dreta, 2 esquerra) que mantenen la garantia de victòria."""
        g = self.game
        head, tail = g.snake[0], g.snake[-1]
        to_tail = self.dist(head, tail)
        to_food = self.dist(head, g.food) if g.food else self.n
        safe = []
        for action, turn in ((0, 0), (1, 1), (2, -1)):
            dx, dy = DIRS[(g.dir + turn) % 4]
            nxt = (head[0] + dx, head[1] + dy)
            if g.is_collision(nxt):
                continue
            d = self.dist(head, nxt)
            if d == 0:
                continue
            # La casella següent del cicle sempre és segura; la resta són dreceres
            if d == 1 or (d < to_tail and d <= to_food):
                safe.append(action)
        return safe

    def cycle_move(self) -> int:
        """L'acció que segueix el cicle exactament (sempre segura)."""
        g = self.game
        head = g.snake[0]
        target = self.cycle[(self.index[head] + 1) % self.n]
        for action, turn in ((0, 0), (1, 1), (2, -1)):
            dx, dy = DIRS[(g.dir + turn) % 4]
            if (head[0] + dx, head[1] + dy) == target:
                return action
        raise RuntimeError("El cicle demana girar 180°: la serp no està alineada amb el cicle.")

    def choose(self, preferences: list[int]) -> int:
        """Tria la primera acció preferida per la IA que sigui segura."""
        safe = self.safe_moves()
        for action in preferences:
            if action in safe:
                return action
        return self.cycle_move()
