from gym.envs.registration import register

register(id = 'PuddleWorld-v1', entry_point='grid_world.envs:PuddleWorldv1', nondeterministic = True)
register(id = 'PWP-v1', entry_point='grid_world.envs:PuddleWorldPrint', nondeterministic = True)
register(id = 'PuddleWorld-v2', entry_point='grid_world.envs:PuddleWorldv2', nondeterministic = True)
register(id = 'PuddleWorld-v3', entry_point='grid_world.envs:PuddleWorldv3', nondeterministic = True)
