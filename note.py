import mujoco
import jax
import brax
from jax import numpy as jp
from brax.io import mjcf
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

# def mj2brax(path, device='gpu', debug=False):
#     mj = mujoco.MjModel.from_xml_path(path)
#     if debug:
#         from jax.lib import xla_bridge
#         assert xla_bridge.get_backend().platform == device, f"jax backend is not {device}"   
#         # change the intergrator from implicit to euler
#         if mj.opt.integrator != 0:
#             print("Brax 目前只支持Euler intergrator")
#             mj.opt.integrator = 0
#         # Barx 目前只支持球形碰撞体, 令mj.geom_contype[i]和mj.geom_conaffinity[i]=0
#         cylinder_flag = False
#         cylinder_idx = []
#         for i, typ in enumerate(mj.geom_type):
#             if typ == 5: # cylinder
#                 cylinder_flag = True
#                 mj.geom_contype[i] = 0
#                 mj.geom_conaffinity[i] = 0
#         if cylinder_flag:
#             print("Brax 目前只支持球形碰撞体, 令mj.geom_contype[i]和mj.geom_conaffinity[i]=0")
#             print("     请检查下列索引的geom: ", cylinder_idx)

#     return mjcf.load_model(mj)
# model = mj2brax(path = f'/work3/s213120/whipping_brax/xml/scene.xml')
# model = mj2brax(path = f'env/whip.xml')

import IPython
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
path = f'env/whip.xml'
mj = mujoco.MjModel.from_xml_path(path)
sys = mjcf.load_model(mj)
IPython.embed()

# sys.dt
# sys.actuator
# sys.init_q          # initial joint position. Note the free joint will have another 1 dim for the quaternion (1, 0, 0, 0)
# sys.link_types      # return a string and every char represents a link type
#                     # eg: "f" for free joint, no parent link
# sys.link_names      # return a list of string, each string represents the link name
# sys.link_parents    # return a list of int, each int represents the parent link index

# sys.num_links()
# sys.dof_link()
# sys.dof_ranges()
# sys.q_idx()
# sys.q_size()
# sys.qd_idx()
# sys.qd_size()
# sys.act_size()

mocap = jp.load('env/mocap.npy')
from brax import kinematics
q = sys.init_q
qd = jp.zeros(sys.qd_size())
x, xd = kinematics.forward(sys, q, qd)