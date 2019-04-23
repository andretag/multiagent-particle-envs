import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        """Define two agents and two targets"""
        world = World()
        world.dim_c = 2  # Communication
        world.collaborative = True

        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        world.landmarks = [Landmark() for i in range(2)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        self.reset_world(world)

        return world

    def reset_world(self, world):
        """Initialize agents with random location.
        But, for targets, ensure enough distance between
        the two targets
        """
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1., 0., 0.])  # Red
            elif i == 1:
                agent.color = np.array([0., 1., 0.])  # Green
            else:
                raise ValueError("Only two agents are supported")
            agent.state.p_pos = np.random.uniform(-1., +1., world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np.random.uniform(-1., +1., world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Check target distances
        while self.check_target_dist(world, th=agent.size * 8) is False:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1., +1., world.dim_p)

    def check_target_dist(self, world, th):
        for i, landmark_i in enumerate(world.landmarks):
            pos_i = landmark_i.state.p_pos
            for j, landmark_j in enumerate(world.landmarks):
                if i != j:
                    pos_j = landmark_j.state.p_pos
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist <= th:
                        return False
        return True

    def reward(self, agent, world):
        """Agents are rewarded based on minimum agent distance 
        to each landmark
        """
        reward = 0.
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            reward -= min(dists)

        return reward

    def observation(self, agent, world):
        """
        For each agent, observation consists of:
        [agent vel, agent pos, target relative pos, agent relateive pos, comm]
        """
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
