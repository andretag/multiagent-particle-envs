## Additional details to use domains
To use the push box domains in your code, please consider the following code:

```
def make_env(args):
    from multiagent.environment import MultiAgentEnv

    # NOTE Modified from: https://github.com/openai/maddpg/blob/master/experiments/train.py
    scenario = scenarios.load(args.env_name + ".py").Scenario()
    world = scenario.make_world()  # Optionally put mode for two box push domain
    done_callback = None
    
    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=done_callback)

    return env
```
