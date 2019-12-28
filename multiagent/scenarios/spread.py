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

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # TODO Same initial pos
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                agent.color = np.array([0.0, 1.0, 0.0])
            else:
                raise NotImplementedError()
            agent.state.p_pos = np.array([0., 0.])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np.random.uniform(-1., +1., world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        while self.check_landmark_dist(world, th=agent.size * 2.) is False:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def check_landmark_dist(self, world, th):
        for i, landmark_i in enumerate(world.landmarks):
            pos_i = landmark_i.state.p_pos
            for j, landmark_j in enumerate(world.landmarks):
                if i != j:
                    pos_j = landmark_j.state.p_pos
                    dist = np.sqrt(np.sum(np.square(pos_i - pos_j)))
                    if dist <= th:
                        return False
        return True

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            raise ValueError("remove")
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

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
