import gym

env = gym.make("MontezumaRevenge-v4")
s = env.reset()
ob, re, done, info = env.step(0)

print(re)
print(info)