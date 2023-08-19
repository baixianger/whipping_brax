import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
from jax import numpy as jp
from brax.io import mjcf
from brax.generalized import pipeline
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

path = f'env/whip_0_slim_dummy_joint.xml'
sys = mjcf.load(path)
print(list(zip(sys.link_names, sys.link_parents, sys.link_types, sys.dof_ranges())))

act = jp.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
q = sys.init_q
qd = jp.zeros(sys.qd_size())
state = jax.jit(pipeline.init)(sys, q, qd)

xpos = []
for i in range(100):
    state = jax.jit(pipeline.step)(sys, state, act)
    xpos.append(state.x.pos)

print(xpos[0].all == None)
