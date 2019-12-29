import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        # Set any world properties first
        world = World()
        world.dim_c = args.n_agent
        world.collaborative = True

        # Add agents
        world.agents = [Agent() for i in range(args.n_agent)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.10

        # Add landmarks
        world.landmarks = [Landmark() for i in range(args.n_agent)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False 

        # Create dummy task
        task = [np.array([0., 0.]) for _ in range(args.n_agent)]
        self.reset_world(world, task)
        return world

    def reset_world(self, world, task):
        for i_agent, agent in enumerate(world.agents):
            if i_agent == 0:
                agent.color = np.array([1.0, 0.0, 0.0])
            elif i_agent == 1:
                agent.color = np.array([0.0, 1.0, 0.0])
            else:
                raise NotImplementedError()
            agent.state.p_pos = np.array([0., 0.])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i_landmark, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = task[i_landmark]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        reward = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            reward -= min(dists)

        return reward

    def observation(self, agent, world):
        agent_poses = []
        for agent in world.agents:
            agent_poses.append(agent.state.p_pos)

        agent_vels = []
        for agent in world.agents:
            agent_vels.append(agent.state.p_vel)

        entity_poses = []
        for entity in world.landmarks:
            entity_poses.append(entity.state.p_pos)

        return np.concatenate(agent_poses + agent_vels + entity_poses)
