"""Import SO_101_SML URDF -> USD (Isaac Sim 6.1), collision mesh đầy đủ (convex decomposition).

Chạy: <ISAAC_SIM_DIR>/python.sh isaac/import_so101.py   (từ gốc repo)
"""
import argparse
import os
import shutil

from isaacsim import SimulationApp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--urdf", default=os.path.join(REPO, "description", "SO_101_SML.urdf"))
parser.add_argument("--out", default=os.path.join(REPO, "isaac", "usd"), help="thư mục xuất USD")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

app = SimulationApp({"headless": True, "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False})

import omni.kit.app

ext_manager = omni.kit.app.get_app().get_extension_manager()
for ext in ("omni.scene.optimizer.core", "isaacsim.robot.schema", "isaacsim.asset.importer.urdf"):
    ext_manager.set_extension_enabled_immediate(ext, True)

from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
import math

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

# Thông số servo STS3215 theo SO-ARM100/Simulation/SO101/so101_new_calib.xml
KP = 998.22  # N·m/rad
KV = 2.731  # N·m·s/rad (actuator) ; damping thụ động 0.6 nằm trong URDF <dynamics>
ARMATURE = 0.028  # kg·m²

config = URDFImporterConfig(
    urdf_path=os.path.abspath(args.urdf),
    usd_path=os.path.abspath(args.out),
    merge_fixed_joints=False,
    fix_base=True,
    collision_from_visuals=False,  # dùng mesh <collision> của URDF (35 part)
    allow_self_collision=False,
    # mesh trong URDF có dạng package://assets/xxx.stl
    ros_package_paths=[{"name": "assets", "path": os.path.join(os.path.dirname(os.path.abspath(args.urdf)), "assets")}],
    robot_type="Manipulator",
    joint_drive_type="force",
    joint_target_type="position",
    override_joint_stiffness=KP,
    override_joint_damping=KV,
)
# Xóa output cũ: transformer có thể bỏ qua thư mục đã tồn tại (theo hướng dẫn Isaac Sim)
stale = os.path.join(config.usd_path, os.path.splitext(os.path.basename(config.urdf_path))[0])
if os.path.isdir(stale):
    shutil.rmtree(stale)
out_usd = URDFImporter(config).import_urdf()
print("OUT_USD", out_usd)

# ---- Hậu xử lý (converter chỉ ghi damping/friction cho Newton/MuJoCo, collision mặc định convexHull) ----
DEG = math.pi / 180.0
PASSIVE_DAMPING = 0.60  # N·m·s/rad (URDF <dynamics damping>)
FRICTION = 0.052  # N·m (URDF <dynamics friction> = MuJoCo frictionloss)
MAX_VEL = 10.0  # rad/s (URDF <limit velocity>)
pkg_dir = os.path.dirname(out_usd)
robot_name = os.path.splitext(os.path.basename(out_usd))[0]

# 1) Variant mặc định = physx (nếu không, mở file sẽ không có vật lý)
root_layer = Sdf.Layer.FindOrOpen(out_usd)
root_layer.GetPrimAtPath(f"/{robot_name}").variantSelections["Physics"] = "physx"
root_layer.Save()

# 2) Thông số khớp cho PhysX: armature, Coulomb + viscous friction, giới hạn vận tốc (đơn vị góc của USD = độ)
physx_stage = Usd.Stage.Open(os.path.join(pkg_dir, "payloads", "Physics", "physx.usda"))
for prim in physx_stage.TraverseAll():  # prim gốc là "over" -> Traverse() bỏ qua
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    # PhysxJointAxisAPI không có Python wrapper trong bản 6.1 -> apply theo tên (giống isaacsim.robot_setup.sysid)
    prim.ApplyAPI("PhysxJointAxisAPI", "angular")
    for attr, value in (
        ("armature", ARMATURE),
        ("staticFrictionEffort", FRICTION),
        ("dynamicFrictionEffort", FRICTION),
        ("viscousFrictionCoefficient", PASSIVE_DAMPING * DEG),
        ("maxJointVelocity", MAX_VEL / DEG),
    ):
        prim.CreateAttribute(f"physxJointAxis:angular:{attr}", Sdf.ValueTypeNames.Float, custom=False).Set(value)
    print("JOINT", prim.GetName(), "armature/friction/viscous/maxvel set")
physx_stage.GetRootLayer().Save()

# 3) Collision đầy đủ: convex decomposition cho toàn bộ mesh collision (giữ phần lõm của part)
inst_stage = Usd.Stage.Open(os.path.join(pkg_dir, "payloads", "instances.usda"))
n = 0
for prim in Usd.PrimRange(inst_stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        n += 1
inst_stage.GetRootLayer().Save()
print("COLLIDERS convexDecomposition:", n)
app.close()
