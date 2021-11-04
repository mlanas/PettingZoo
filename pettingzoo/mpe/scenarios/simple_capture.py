import numpy as np

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, n_agents=3, n_landmarks=8):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = n_agents
        num_landmarks = n_landmarks
        world.collaborative = True
        world.landmark_capture = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        self.empty_pos = np.full_like(world.agents[0].state.p_pos, 5)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.alive_landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.alive_agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.alive_agents:
                if a is agent: continue
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
        if agent.collide:
            for a in world.alive_agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 1
            for l in world.alive_landmarks:
                if self.is_collision(l, agent):
                    rew += 1
        return rew

    def global_reward(self, world):
        rew = 0
        for l in world.alive_landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.alive_agents]
            rew -= min(dists)
        if rew == 0: rew = 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.entities: # agents + landmarks
            if entity is agent: continue
            relative_pos = entity.state.p_pos - agent.state.p_pos if entity.state.alive else self.empty_pos # the relative position is zero if the entity is dead (death masking)
            entity_pos.append(relative_pos)
        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + entity_pos)
