import os
import torch
import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

MODEL_DIR = "./model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model.pth")          # только веса (state_dict)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")  # веса + optimizer + record + n_games
RECORD_PATH = os.path.join(MODEL_DIR, "record.txt")          # просто число (для удобства)


def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


def load_record_txt() -> int:
    try:
        with open(RECORD_PATH, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return 0


def save_record_txt(record: int) -> None:
    with open(RECORD_PATH, "w", encoding="utf-8") as f:
        f.write(str(int(record)))


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.global_record = 0
        self._try_load_checkpoint_or_weights()

    def _try_load_checkpoint_or_weights(self):
        ensure_model_dir()

        # 1) Основной вариант: checkpoint (веса + оптимайзер + рекорд + n_games)
        if os.path.exists(CHECKPOINT_PATH):
            data = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))

            self.model.load_state_dict(data["model_state"])
            if "optimizer_state" in data and data["optimizer_state"] is not None:
                self.trainer.optimizer.load_state_dict(data["optimizer_state"])

            self.n_games = int(data.get("n_games", 0))
            self.global_record = int(data.get("record", 0))

            print(f"[Agent] Loaded CHECKPOINT: {CHECKPOINT_PATH}")
            print(f"[Agent] global_record={self.global_record}, n_games={self.n_games}")
            return

        # 2) Миграция со старого формата: только веса + record.txt (если есть)
        if os.path.exists(WEIGHTS_PATH):
            state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)

            self.global_record = load_record_txt()  # если файла нет — будет 0
            self.n_games = 0

            print(f"[Agent] Loaded WEIGHTS only: {WEIGHTS_PATH}")
            print(f"[Agent] global_record(from record.txt)={self.global_record}, n_games reset to 0")
            return

        print("[Agent] No saved model found. Starting from scratch.")
        self.global_record = 0
        self.n_games = 0

    def save_if_global_record_beaten(self, score: int):
        """
        Перезаписывает файлы ТОЛЬКО если побит ГЛОБАЛЬНЫЙ рекорд.
        """
        if score <= self.global_record:
            return False

        self.global_record = int(score)
        ensure_model_dir()

        # Сохраняем полный чекпоинт (лучший вариант для продолжения обучения)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.trainer.optimizer.state_dict(),
                "record": self.global_record,
                "n_games": int(self.n_games),
            },
            CHECKPOINT_PATH,
        )

        # Дополнительно сохраняем только веса (на всякий)
        torch.save(self.model.state_dict(), WEIGHTS_PATH)

        # И рекорд текстом — удобно смотреть
        save_record_txt(self.global_record)

        print(f"[Agent] ✅ GLOBAL RECORD BEATEN! Saved checkpoint. record={self.global_record}")
        return True

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.save_if_global_record_beaten(score)

            print(
                "TRAIN | Game", agent.n_games,
                "| Score", score,
                "| GlobalRecord:", agent.global_record
            )

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
