"""Kiểm thử SO_101_SML trong Isaac Sim: đứng vững, bám target, collision (thả hộp lên robot), chụp ảnh.

Chạy: <ISAAC_SIM_DIR>/python.sh isaac/test_so101.py   (từ gốc repo)
GPU render phải là GPU chạy được RTX/Vulkan (server lab: GPU 0).
"""
import argparse
import os

from isaacsim import SimulationApp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=os.path.join(REPO, "isaac", "usd", "SO_101_SML", "SO_101_SML.usda"))
parser.add_argument("--out", default=os.path.join(REPO, "isaac", "test_out"), help="thư mục lưu ảnh chụp")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

app = SimulationApp({"headless": True, "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
                     "width": 1280, "height": 720})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd, UsdGeom, UsdPhysics

USD = os.path.abspath(args.usd)
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)

stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/Light").set_intensities(1500)
stage_utils.add_reference_to_stage(usd_path=USD, path="/World/so101")
stage = omni.usd.get_context().get_stage()

# Thống kê collider thực tế trên stage
n_col, approx = 0, {}
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/so101"), Usd.TraverseInstanceProxies()):
    if p.HasAPI(UsdPhysics.CollisionAPI):
        n_col += 1
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            a = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
            approx[a] = approx.get(a, 0) + 1
print(f"RESULT colliders={n_col} approx={approx}")

bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"]).ComputeWorldBound(
    stage.GetPrimAtPath("/World/so101")).ComputeAlignedRange()
lo, hi = np.array(bbox.GetMin()), np.array(bbox.GetMax())
print(f"RESULT robot_bbox min={lo.round(3)} max={hi.round(3)}")

arm = Articulation("/World/so101")
SimulationManager.set_physics_dt(1.0 / 120.0)


def step(n):
    for _ in range(n):
        SimulationManager.step()
        RenderingManager.render()
        app.update()


def capture(name):
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    vp = get_active_viewport()
    capture_viewport_to_file(vp, os.path.join(OUT, name))
    for _ in range(10):
        app.update()


center = (lo + hi) / 2
span = float(np.max(hi - lo))
set_camera_view(eye=list(center + np.array([span * 1.6, span * 1.6, span * 1.0])), target=list(center),
                camera_prim_path="/OmniverseKit_Persp")

app_utils.play()
app.update()
names = arm.dof_names
print("RESULT dof_names", names)

# 1) Đứng vững: giữ target = 0 trong 2 s dưới trọng lực
arm.set_dof_position_targets(np.zeros((1, len(names))))
step(240)
q0 = arm.get_dof_positions().numpy()[0]
print("RESULT hold_zero_err_deg", np.rad2deg(np.abs(q0)).round(2).tolist())
capture("01_home.png")

# 2) Bám target (trong giới hạn, wrist_roll âm vì limit [-360°, 0°])
target_deg = {"shoulder_pan": 30, "shoulder_lift": -30, "elbow_flex": 45, "wrist_flex": 30,
              "wrist_roll": -90, "gripper": 45}
target = np.deg2rad([target_deg.get(n, 0) for n in names])
arm.set_dof_position_targets(target[None])
step(360)
q1 = arm.get_dof_positions().numpy()[0]
print("RESULT track_target_deg", np.rad2deg(target).round(1).tolist())
print("RESULT track_actual_deg", np.rad2deg(q1).round(2).tolist())
capture("02_pose.png")

# 3) Collision: thả khối hộp 2 cm từ trên xuống ngay trên robot
arm.set_dof_position_targets(np.zeros((1, len(names))))
step(240)
drop_xy = center[:2]
cube = Cube("/World/cube", sizes=0.02, positions=[[drop_xy[0], drop_xy[1], hi[2] + 0.10]])
UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/World/cube"))
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/cube"))
UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/cube")).CreateMassAttr(0.01)
app_utils.stop()
app_utils.play()
app.update()
arm.set_dof_position_targets(np.zeros((1, len(names))))
step(360)
cube_z = RigidPrim("/World/cube").get_world_poses()[0].numpy()[0][2]
print(f"RESULT cube_drop_from={hi[2] + 0.10:.3f} cube_final_z={cube_z:.4f} (ground would be ~0.010)")
capture("03_collision.png")

ok = np.all(np.isfinite(q1))
print("RESULT finite_state", bool(ok))
app_utils.stop()
app.close()
