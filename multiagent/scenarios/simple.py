import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        """Define one agents and one target.
        Note that world.golas are used only for hierarchical RL
        visualization only
        """
        world = World()

        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True

        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        self.reset_world(world)

        return world

    def reset_world(self, world):
        """Define random properties for agents, box, and target.
        One agent and one target are randomly initialized.
        However, they are initialized to ensure sufficient distance between the two
        """
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            while self.check_distance(world.agents, landmark.state.p_pos):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

    def check_distance(self, agents, landmark_p_pos):
        th = 1.
        for agent in agents:
            if np.linalg.norm(agent.state.p_pos - landmark_p_pos) < th:
                return True
            else:
                return False

    def reward(self, agent, world):
        return -np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)

    def observation(self, agent, world):
        """For each agent, observation consists of:
        [agent velocity, agent pos, target pos]
        """
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)
