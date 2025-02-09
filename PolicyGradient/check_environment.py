import gymnasium as gym
import time


def main():
    # Create the ALE Pong-v5 environment with a human render mode.
    env = gym.make("ALE/Pong-v5", render_mode="human")

    # Reset the environment; Gymnasium returns (obs, info)
    observation, info = env.reset(seed=42)

    done = False
    step_count = 0
    print("Starting ALE/Pong-v5 environment...")

    try:
        while not done:
            # Choose a random action
            action = env.action_space.sample()

            # Step the environment; note the additional return values for termination
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            print(
                f"Step {step_count}: Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}")
            step_count += 1

            # Optional: slow down the loop so rendering is visible
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
