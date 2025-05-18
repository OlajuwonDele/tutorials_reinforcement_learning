import gymnasium
import gymnasium_env

env = gymnasium.make('gymnasium_env/GridWorld-v0', render_mode="human")

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step: Obs={obs}, Reward={reward}, Done={done}, Info={info}")

env.close()
