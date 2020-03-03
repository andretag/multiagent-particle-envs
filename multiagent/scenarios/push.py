import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.collaborative = True
        world.clip_positions = True

        world.agents = [Agent() for i in range(args.n_agent)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1  # Radius

        self.boxes = [Landmark() for _ in range(1)]
        for i, box in enumerate(self.boxes):
            box.name = 'box %d' % i
            box.collide = True
            box.movable = True
            box.size = 0.35
            box.initial_mass = 9.
            box.index = i
            world.landmarks.append(box)

        self.targets = [Landmark() for _ in range(1)]
        for i, target in enumerate(self.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.movable = False
            target.size = 0.05
            target.index = i
            world.landmarks.append(target)

        self.reset_world(world)
        return world

    def reset_world(self, world, task=None):
        """Define random properties for agents, box, and targets.
        Two agents are randomly initialized.
        The box and the left target are initialized at the same location on the left side
        """
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1., 0., 0.])  # Red
                agent.state.p_pos = np.array([0.5, 0.5])  # Box
            elif i == 1:
                agent.color = np.array([0., 1., 0.])  # Blue
                agent.state.p_pos = np.array([0.5, -0.5])  # Box
            else:
                raise NotImplementedError()
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)

            if "box" in landmark.name and landmark.index == 0:
                landmark.state.p_pos = np.array([-0.15, 0.0])  # Box
            elif "target" in landmark.name and landmark.index == 0:
                landmark.state.p_pos = np.array([-0.85, 0.0])  # Left target
            else:
                raise ValueError()

    def reward(self, agent, world):
        """Reward is defined to be large if distance between box and target0 is minimized"""
        for i, landmark in enumerate(world.landmarks):
            if "box" in landmark.name and landmark.index == 0:
                box0 = landmark
            elif "target" in landmark.name and landmark.index == 0:
                target0 = landmark
            else:
                raise ValueError()

        dist = np.sum(np.square(box0.state.p_pos - target0.state.p_pos))
        return -dist

    def observation(self, agent, world):
        """For each agent, observation consists of:
        [agent poses, box pos, target pos]
        """
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)
        assert len(entity_pos) == len(self.boxes) + len(self.targets)

        agent_pos = []
        for agent in world.agents:
            agent_pos.append(agent.state.p_pos)

        agent_vel = []
        for agent in world.agents:
            agent_vel.append(agent.state.p_vel)

        return np.concatenate(entity_pos + agent_pos + agent_vel)
