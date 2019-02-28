import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, mode=0):
        """
        mode0:
        - Random agent loc
        - Box starts at left side
        - Target at right side

        mode1:
        - Random agent loc
        - Box starts at right side
        - Target at left side

        mode2:
        - Random agent loc
        - Box at center
        - Target at random
        """
        assert mode >= 0
        assert mode <= 2
        self.mode = mode

        world = World()

        # add agents
        self.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(self.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.i = i
            world.agents.append(agent)

        # add boxes
        n_box = 1  # One box and pushing to left
        self.boxes = [Landmark() for _ in range(n_box)]
        for i, box in enumerate(self.boxes):
            box.name = 'box %d' % i
            box.collide = True
            box.movable = True
            box.size = 0.25
            box.initial_mass = 5.
            box.index = i
            world.landmarks.append(box)

        # add targets
        self.targets = [Landmark() for _ in range(n_box)]
        for i, target in enumerate(self.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.movable = False
            target.size = 0.05
            target.index = i
            world.landmarks.append(target)

        # make initial conditions
        self.reset_world(world)

        self.timestep = 0.
        
        return world

    def reset_world(self, world):
        # random properties for agents
        option = np.random.randint(low=0, high=1)
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1.0, 0.0, 0.0])
                if option == 0:
                    agent.state.p_pos = np.array([0.40, 0.40])
                elif option == 1:
                    agent.state.p_pos = np.array([0.40, -0.40])
                else:
                    raise ValueError()
            elif i == 1:
                agent.color = np.array([0.0, 1.0, 0.0])
                if option == 0:
                    agent.state.p_pos = np.array([0.40, -0.40])
                elif option == 1:
                    agent.state.p_pos = np.array([0.40, 0.40])
                else:
                    raise ValueError()
            else:
                raise NotImplementedError()

            # agent.state.p_pos = np.random.uniform(-0.4, +0.4, world.dim_p)
            assert self.check_overlap(agent.state.p_pos, i, world) is True
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)

            if "box" in landmark.name and landmark.index == 0:
                if self.mode == 0:
                    random_x, random_y = 0., 0.
                elif self.mode == 1:
                    random_x = float(np.random.uniform(low=-0.85, high=-0.50, size=1))
                    random_y = float(np.random.uniform(low=-0.15, high=0.15, size=1))
                elif self.mode == 2:
                    random_x = float(np.random.uniform(low=-0.15, high=0.15, size=1))
                    random_y = float(np.random.uniform(low=-0.15, high=0.15, size=1))
                else:
                    raise ValueError()
            elif "target" in landmark.name and landmark.index == 0:
                if self.mode == 0:
                    random_x, random_y = -0.85, 0.
                elif self.mode == 1:
                    random_x = float(np.random.uniform(low=0.50, high=0.85, size=1))
                    random_y = float(np.random.uniform(low=-0.85, high=0.85, size=1))
                elif self.mode == 2:
                    random_x = float(np.random.uniform(low=-0.85, high=0.85, size=1))
                    random_y = float(np.random.uniform(low=-0.85, high=0.85, size=1))
                else:
                    raise ValueError()
            else:
                raise ValueError()
            landmark.state.p_pos = np.array([random_x, random_y])

        self.timestep = 0.

    def check_overlap(self, p_pos, i_agent, world):
        box_radius = self.boxes[0].size
        agent_radius = self.agents[0].size
        box2agent_radius = box_radius + agent_radius
        agent2agent_radius = agent_radius + agent_radius

        if abs(p_pos[0]) < box2agent_radius and abs(p_pos[1]) < box2agent_radius:
            return False
        else:
            if i_agent == 1:
                x_diff = abs(p_pos[0] - world.agents[0].state.p_pos[0])
                y_diff = abs(p_pos[1] - world.agents[0].state.p_pos[1])

                if x_diff < agent2agent_radius and y_diff < agent2agent_radius:
                    return False
                else:
                    return True
            else:
                return True

    def reward(self, agent, world):
        for i, landmark in enumerate(world.landmarks):
            if "box" in landmark.name and landmark.index == 0:
                box0 = landmark
            elif "target" in landmark.name and landmark.index == 0:
                target0 = landmark
            else:
                raise ValueError()

        # Move box0 to target0 (One Box)
        dist = np.sum(np.square(box0.state.p_pos - target0.state.p_pos)) * 50.

        return -dist

    def observation(self, agent, world):
        if agent.i == 0:
            self.timestep += 1

        # get positions of all entities
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        assert len(entity_pos) == len(self.boxes) + len(self.targets)

        # Add other agent position
        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [np.array([self.timestep / 100.])])
