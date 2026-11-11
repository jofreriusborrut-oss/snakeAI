"""Mira com juga la IA entrenada.

Ús:
    python play.py              # finestra gràfica (cal pygame)
    python play.py --text       # al terminal, sense pygame
    python play.py --games 20 --fast   # avalua 20 partides sense dibuixar
    python play.py --segur      # amb el guardià: guanya totes les partides
"""
import argparse
import os
import time

from agent import DQNAgent, get_state
from game import SnakeGame
from guardian import Guardian

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 24


def make_policy(agent, segur):
    """Retorna una funció que, donat el joc, diu quina acció fer."""
    if not segur:
        return lambda game: agent.act(get_state(game), explore=False)
    guardians = {}

    def policy(game):
        g = guardians.get(id(game)) or guardians.setdefault(id(game), Guardian(game))
        return g.choose(agent.ranked_actions(get_state(game)))
    return policy


def result_text(game):
    return "VICTÒRIA! Ha omplert tot el tauler" if game.food is None else f"{game.score} punts"


def play_text(game, policy, delay):
    while True:
        res = game.step(policy(game))
        print("\033[H\033[J" + game.render_text() + f"\nPunts: {game.score}")
        time.sleep(delay)
        if res.done:
            return game.score


def play_pygame(game, policy, fps):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((game.w * CELL, game.h * CELL + 30))
    pygame.display.set_caption("IA del Snake")
    font = pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        res = game.step(policy(game))
        screen.fill((20, 20, 28))
        if game.food:
            pygame.draw.rect(screen, (230, 70, 70), (game.food[0] * CELL, game.food[1] * CELL, CELL, CELL))
        for i, (x, y) in enumerate(game.snake):
            color = (120, 230, 120) if i == 0 else (60, 170, 80)
            pygame.draw.rect(screen, color, (x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2))
        screen.blit(font.render(f"Punts: {game.score}", True, (240, 240, 240)), (8, game.h * CELL + 6))
        pygame.display.flip()
        clock.tick(fps)
        if res.done:
            time.sleep(0.8)
            return game.score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(HERE, "model", "snake_dqn.pt"))
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--text", action="store_true", help="dibuixa al terminal")
    p.add_argument("--fast", action="store_true", help="no dibuixa; només dona resultats")
    p.add_argument("--fps", type=int, default=None,
                   help="moviments per segon (per defecte 15, o 300 amb --segur)")
    p.add_argument("--segur", action="store_true",
                   help="activa el guardià (cicle hamiltonià): la serp guanya sempre")
    args = p.parse_args()

    agent = DQNAgent()
    agent.load(args.model)
    game = SnakeGame(args.size, args.size)
    policy = make_policy(agent, args.segur)
    if args.fps is None:
        args.fps = 300 if args.segur else 15
    scores, wins, t0 = [], 0, time.time()
    for n in range(1, args.games + 1):
        game.reset()
        if args.fast:
            while not game.step(policy(game)).done:
                pass
            score = game.score
        elif args.text:
            score = play_text(game, policy, 1 / args.fps)
        else:
            score = play_pygame(game, policy, args.fps)
        scores.append(score)
        wins += game.food is None
        print(f"Partida {n}: {result_text(game)} ({score} punts, {game.frames} moviments)")
    print(f"Mitjana: {sum(scores) / len(scores):.1f} | Màxim: {max(scores)} | "
          f"Victòries: {wins}/{len(scores)} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
