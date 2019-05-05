import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
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

        # make initial conditions
        self.reset_world(world, mode=0)

        return world

    def reset_world(self, world, mode):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.state.p_pos = np.array([-0.85, -0.85])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
            if mode == 0:
                landmark.state.p_pos = np.array([-0.85, 0.85])
            elif mode == 1:
                landmark.state.p_pos = np.array([0.85, -0.85])
            else:
                print(mode)
                raise ValueError("Invalid mode")
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.goal_reached = False
        self.mode = mode

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        if self.goal_reached is False:
            dist = np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
            if dist < 0.05:
                self.goal_reached = True
                world.landmarks[0].state.p_pos = np.array([0.85, 0.85])

        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)
        return np.concatenate([agent.state.p_pos] + entity_pos)
