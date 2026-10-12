"""Pick & place khoi cube 2 cm bang SO_101_SML trong Isaac Sim 6.1 (headless) + quay video MP4.

Chay (tu goc repo):
    <ISAAC_SIM_DIR>/python.sh isaac/pick_place_cube.py --gpu 0 --out isaac/test_out/pick_place
    them --no-tuning de chay voi thong so mac dinh (khong ap dung tinh chinh gap)

Dieu khien: IK so (isaac/so101_kinematics.py, doc URDF) cho 5 khop canh tay, gap tu tren xuong;
servo giu nguyen thong so that (kp 998.22, effort 2.94 N*m) - ham kep dong toi 0 deg (bao hoa luc nhu servo that).

Tinh chinh gap ap dung (theo robotiq/isaacsim_assets GRIPPER_SIMULATION_GUIDE.md §3):
  - vat: khoi luong > 0, contactOffset 2 mm / restOffset 0.5 mm, linearDamping 0.1,
    solver velocity iterations 4, speculative CCD (§3.3), ma sat xoan (torsionalPatchRadius)
  - ma sat: vat lieu vat ly cho 2 ham kep va cube
  - solver: articulation 64/4 iterations, 2 khau ham kep 64/8, PhysicsScene enableStabilization, 120 Hz, CPU
Ket qua: dong "RESULT ..." trong log + <out>/pick_place_cube.mp4 (luoi 2x2 1920x1080: toan canh, chinh dien
diem gap, chinh dien diem dat, camera gan theo gripper nhin chinh dien khe kep) + anh PNG cac buoc chinh.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

from isaacsim import SimulationApp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=os.path.join(REPO, "isaac", "usd", "SO_101_SML", "SO_101_SML.usda"))
parser.add_argument("--urdf", default=os.path.join(REPO, "description", "SO_101_SML.urdf"))
parser.add_argument("--out", default=os.path.join(REPO, "isaac", "test_out", "pick_place"))
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg")
parser.add_argument("--no-tuning", action="store_true", help="khong ap dung tinh chinh gap (de so sanh)")
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

app = SimulationApp({"headless": True, "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False,
                     "width": 1280, "height": 720})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_kinematics import JOINTS, SO101Kinematics, tf  # noqa: E402

TUNED = not args.no_tuning
PHYSICS_HZ = 120
CUBE = 0.02
CUBE_MASS = 0.010            # kg, PLA in 3D ~1.24 g/cm3 x 8 cm3
PICK = np.array([0.0, -0.26])  # tam cube (m, khung the gioi = khung base_arm)
PLACE = np.array([0.15, -0.21])
TRANSIT_Z = 0.06              # do cao TCP khi di chuyen (gap thang dung: wrist_flex cham -90 deg tu ~0.075)
# TCP "grasp" trong khung gripper_frame_link: ngon chia theo +y, ham dong mo theo +z (0 deg = dong).
# Ham co dinh nho them 6.3 mm qua goc gripper_frame_link theo +y -> tam cube o y = 6.3 + 2 - 10 = -1.7 mm
# de dau ham cach san 2 mm khi gap (cube nam o ~2 cm cuoi cua ngon). Tam cube cach mat trong ham co dinh 1.5 mm.
GRASP_OFFSET = (0.0, -0.0017, 0.0115)
GRIP_OPEN = math.radians(30)   # mat trong ham dong o z >= 39 mm -> du cho cube 2 cm
GRIP_CLOSE = 0.0               # dong het: servo bao hoa 2.94 N*m ep cube vao ham co dinh
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- scene
stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
DistantLight("/World/Light").set_intensities(1500)
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.0)
stage_utils.add_reference_to_stage(usd_path=os.path.abspath(args.usd), path="/World/so101")

# PhysicsScene rieng: 120 Hz, CPU, MBP; stabilization chi khi tinh chinh
scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px_scene.CreateEnableGPUDynamicsAttr(False)
px_scene.CreateBroadphaseTypeAttr("MBP")
px_scene.CreateTimeStepsPerSecondAttr(PHYSICS_HZ)
px_scene.CreateEnableStabilizationAttr(TUNED)

kin = SO101Kinematics(os.path.abspath(args.urdf))
kin.add_tcp("grasp", "gripper_frame_link", tf(p=GRASP_OFFSET))


def find(root, name, need_body=True):
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.GetName() == name and (not need_body or p.HasAPI(UsdPhysics.RigidBodyAPI)):
            return p
    return None


def physics_material(path, static, dynamic):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def bind_physics(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


checks = []
art_root = next((p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/so101")) if p.HasAPI(UsdPhysics.ArticulationRootAPI)), None)
jaw_fixed, jaw_moving = find("/World/so101", "part_56"), find("/World/so101", "moving_jaw_so101_v1")
checks.append((art_root is not None and jaw_fixed is not None and jaw_moving is not None,
               f"articulation={art_root.GetPath() if art_root else None} ham co dinh={jaw_fixed.GetPath() if jaw_fixed else None} "
               f"ham dong={jaw_moving.GetPath() if jaw_moving else None}"))

# cube tren san, tai diem gap
Cube("/World/cube", sizes=CUBE, positions=[[PICK[0], PICK[1], CUBE / 2 + (0.0005 if TUNED else 0.0)]])
cube = stage.GetPrimAtPath("/World/cube")
UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.2, 0.15)])
UsdPhysics.RigidBodyAPI.Apply(cube)
UsdPhysics.CollisionAPI.Apply(cube)

if TUNED:
    # §3.1: vat co khoi luong, offset tiep xuc, giam chan, velocity iterations; §3.3: speculative CCD
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(CUBE_MASS)
    col = PhysxSchema.PhysxCollisionAPI.Apply(cube)
    col.CreateContactOffsetAttr(0.002)
    col.CreateRestOffsetAttr(0.0005)
    col.CreateTorsionalPatchRadiusAttr(0.005)  # ma sat xoan: cube khong xoay quanh truc kep
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube)
    rb.CreateLinearDampingAttr(0.1)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateEnableSpeculativeCCDAttr(True)
    # ma sat: cube PLA, ham kep gia dinh co dem TPU (SO-101 thuong dan dem cao su)
    bind_physics(cube, physics_material("/World/Physics/CubeMaterial", 0.8, 0.7))
    pad = physics_material("/World/Physics/JawPadMaterial", 1.0, 0.9)
    for jaw in (jaw_fixed, jaw_moving):
        bind_physics(jaw, pad)
        jrb = PhysxSchema.PhysxRigidBodyAPI.Apply(jaw)
        jrb.CreateSolverPositionIterationCountAttr(64)
        jrb.CreateSolverVelocityIterationCountAttr(8)
    if art_root is not None:
        pa = PhysxSchema.PhysxArticulationAPI.Apply(art_root)
        pa.CreateSolverPositionIterationCountAttr(64)
        pa.CreateSolverVelocityIterationCountAttr(4)
    checks.append((True, "tinh chinh: cube 10 g, contact/rest 2/0.5 mm, damping 0.1, iters 16/4, speculative CCD, "
                         "torsional 5 mm, ma sat cube 0.8/0.7 - ham 1.0/0.9, articulation 64/4, ham 64/8, stabilization"))
else:
    UsdPhysics.MassAPI.Apply(cube).CreateMassAttr(CUBE_MASS)  # van can khoi luong > 0 (thieu thi xuyen qua moi thu)
    checks.append((True, "KHONG tinh chinh: chi co khoi luong cube 10 g, con lai mac dinh PhysX"))

# ---- camera quay video: luoi 2x2 (moi o 960x540)
TILE_W, TILE_H = 960, 540


def make_camera(path, focal, xform=None, eye=None, target=None, up=(0, 0, 1)):
    cam = UsdGeom.Camera.Define(stage, path)
    if xform is None:
        xform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(xform)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * TILE_H / TILE_W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 20.0))
    return path


# Luc gap/dat (yaw 0): phuong kep z_g = +X the gioi (ham co dinh ben -X, ham dong ben +X), ngon chia xuong.
# Camera "chinh dien" nhin doc +Y (vuong goc phuong kep) ngang tam cube -> thay 2 ham ep 2 ben cube.
FRONT_DIST, FRONT_Z = 0.13, 0.018
CAMERAS = [  # (duong dan, nhan tren video)
    (make_camera("/World/Cam_Overview", 18.0, eye=(0.42, -0.62, 0.36), target=(0.06, -0.20, 0.04)), "Toan canh"),
    (make_camera("/World/Cam_FrontPick", 26.0, eye=(PICK[0], PICK[1] - FRONT_DIST, FRONT_Z),
                 target=(PICK[0], PICK[1], FRONT_Z)), "Chinh dien - diem gap"),
    (make_camera("/World/Cam_FrontPlace", 26.0, eye=(PLACE[0], PLACE[1] - FRONT_DIST, FRONT_Z),
                 target=(PLACE[0], PLACE[1], FRONT_Z)), "Chinh dien - diem dat"),
]
# Camera gan theo gripper (con cua khau ham co dinh): dat o +x_g cua gripper_frame_link, nhin nguoc -x_g vao khe
# giua 2 ham, anh "len" = phia co tay (-y_g). Truc camera USD: nhin -Z, len +Y -> X=z_g, Y=-y_g, Z=x_g.
q_zero = np.zeros(6)
T_link_G = np.linalg.inv(kin.fk(q_zero, "part_56")) @ kin.fk(q_zero, "gripper_frame_link")
T_G_cam = tf(np.column_stack([[0, 0, 1], [0, -1, 0], [1, 0, 0]]), (0.12, -0.012, 0.008))
T_link_cam = T_link_G @ T_G_cam
if jaw_fixed is not None:
    follow = make_camera(f"{jaw_fixed.GetPath()}/Cam_GripperFront", 22.0,
                         xform=Gf.Matrix4d(*T_link_cam.T.flatten().tolist()))
    CAMERAS.append((follow, "Gan theo gripper - chinh dien khe kep"))

for ok, msg in checks:
    log(f"CHECK {'PASS' if ok else 'FAIL'} {msg}")

# ---------------------------------------------------------------- ke hoach chuyen dong (IK)


def down_pose(p, yaw=0.0):
    """TCP: ngon chia xuong (-z), phuong kep (z_g) nam ngang theo goc yaw."""
    y = np.array([0, 0, -1.0])
    z = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return tf(np.column_stack([np.cross(y, z), y, z]), p)


IK_SEEDS = [np.radians(s) for s in ([0, -10, 0, -80, -90, 0], [0, -15, -15, -60, -90, 0], [0, -30, 60, 60, -90, 0])]


def solve(p, q_seed, yaw=0.0):
    goal = down_pose(np.asarray(p, float), yaw)
    q, ep, er = kin.ik(goal, q_seed, "grasp")  # uu tien nghiem gan q_seed (chuyen dong lien tuc)
    for seed in IK_SEEDS:
        if ep < 1e-3 and er < math.radians(0.5):
            break
        s = seed.copy()
        s[0] = math.atan2(p[0], -(p[1] - kin.fk(np.zeros(6), "part_31")[1, 3]))  # huong shoulder_pan ve diem dich
        q, ep, er = kin.ik(goal, s, "grasp")
    if ep > 1e-3 or er > math.radians(0.5):
        raise RuntimeError(f"IK khong toi duoc {p}: loi {ep * 1000:.1f} mm {math.degrees(er):.1f} deg")
    return q


cz = CUBE / 2
q_ready = solve([PICK[0], PICK[1], TRANSIT_Z + 0.005], IK_SEEDS[0])
q_pre = solve([PICK[0], PICK[1], TRANSIT_Z], q_ready)
q_grasp = solve([PICK[0], PICK[1], cz], q_pre)
q_lift = q_pre.copy()
q_place_pre = solve([PLACE[0], PLACE[1], TRANSIT_Z], q_lift)
q_place = solve([PLACE[0], PLACE[1], cz + 0.001], q_place_pre)
log("RESULT ik_ok ready/pre/grasp/place (deg): " + " | ".join(
    str(np.degrees(q[:5]).round(1).tolist()) for q in (q_ready, q_pre, q_grasp, q_place)))


def cartesian(p0, p1, q_seed, n=12):
    """Doan thang Cartesian (MoveL) -> danh sach q."""
    out, q = [], q_seed
    for s in np.linspace(0, 1, n + 1)[1:]:
        q = solve(np.asarray(p0) + s * (np.asarray(p1) - np.asarray(p0)), q)
        out.append(q)
    return out


# Moi buoc: (ten, danh sach q dich cho canh tay, goc ham kep, thoi gian s)
plan = [
    ("ve tu the san sang", [q_ready], GRIP_OPEN, 2.0),
    ("tren cube", [q_pre], GRIP_OPEN, 1.0),
    ("ha xuong (MoveL)", cartesian([*PICK, TRANSIT_Z], [*PICK, cz], q_pre), GRIP_OPEN, 2.0),
    ("kep", [q_grasp], GRIP_CLOSE, 1.2),
    ("nang (MoveL)", cartesian([*PICK, cz], [*PICK, TRANSIT_Z], q_grasp), GRIP_CLOSE, 2.0),
    ("di chuyen", [q_place_pre], GRIP_CLOSE, 2.5),
    ("ha xuong (MoveL)", cartesian([*PLACE, TRANSIT_Z], [*PLACE, cz + 0.001], q_place_pre), GRIP_CLOSE, 2.0),
    ("nha", [q_place], GRIP_OPEN, 1.0),
    ("rut len (MoveL)", cartesian([*PLACE, cz + 0.001], [*PLACE, TRANSIT_Z], q_place), GRIP_OPEN, 1.5),
    ("ve tu the san sang", [q_ready], GRIP_OPEN, 2.0),
]

# ---------------------------------------------------------------- mo phong + quay video
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(1.0 / PHYSICS_HZ, physics_scene="/World/PhysicsScene")
annots = []
for cam_path, label in CAMERAS:
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rep.create.render_product(cam_path, (TILE_W, TILE_H))])
    annots.append((a, label))
VIDEO_W, VIDEO_H = TILE_W * 2, TILE_H * 2

app_utils.play()
RenderingManager.render()
arm = Articulation("/World/so101")
names = list(arm.dof_names)
idx = [names.index(n) for n in JOINTS]
log(f"RESULT dof_names {names}")
cube_rb = RigidPrim("/World/cube")


def set_q(q_arm, grip):
    tgt = np.zeros(len(names))
    for i, n in enumerate(JOINTS):
        tgt[names.index(n)] = grip if n == "gripper" else q_arm[i]
    arm.set_dof_position_targets(tgt[None])


def joint_q():
    q = arm.get_dof_positions().numpy()[0]
    return np.array([q[i] for i in idx])


def cube_pose():
    p, quat = cube_rb.get_world_poses()
    p, quat = p.numpy()[0], quat.numpy()[0]  # quat (w, x, y, z)
    w, x, y, z = quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return p, R


# dua robot ve tu the san sang ngay tu dau (target + trang thai) roi de cube nam yen
q_home = np.zeros(len(names))
for i, n in enumerate(JOINTS):
    q_home[names.index(n)] = GRIP_OPEN if n == "gripper" else q_ready[i]
arm.set_dof_positions(q_home[None])
set_q(q_ready, GRIP_OPEN)
SimulationManager.step(steps=PHYSICS_HZ)

ff = subprocess.Popen([args.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-crf", "20", "-preset", "veryfast", os.path.join(OUT, "pick_place_cube.mp4")], stdin=subprocess.PIPE)
try:
    font = ImageFont.load_default(size=22)
    font_s = ImageFont.load_default(size=16)
except TypeError:
    font = font_s = ImageFont.load_default()

steps_per_frame = PHYSICS_HZ // args.fps
t_sim = 0.0
frames = 0
metrics = {"slip_mm": 0.0, "tilt_hold_deg": 0.0, "grip_deg": None, "jitter_mm": []}
hold_ref = None  # vi tri cube trong khung TCP luc vua kep xong


def grab(phase, info):
    global frames
    RenderingManager.render()
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    for k, (a, label) in enumerate(annots):
        data = a.get_data()
        x0, y0 = (k % 2) * TILE_W, (k // 2) * TILE_H
        if data is not None and getattr(data, "size", 0):
            img.paste(Image.fromarray(np.asarray(data)[..., :3].astype(np.uint8)), (x0, y0))
        if k:  # o 0 (toan canh) co khung thong tin rieng
            d.rectangle([x0, y0 + TILE_H - 34, x0 + 12 + 11 * len(label), y0 + TILE_H], fill=(0, 0, 0))
            d.text((x0 + 8, y0 + TILE_H - 30), label, font=font, fill=(255, 255, 255))
    d.line([(TILE_W, 0), (TILE_W, VIDEO_H)], fill=(255, 255, 255), width=2)
    d.line([(0, TILE_H), (VIDEO_W, TILE_H)], fill=(255, 255, 255), width=2)
    d.rectangle([0, 0, 560, 92], fill=(0, 0, 0))
    d.text((12, 8), f"SO-101 SML | cube 2 cm | {'DA tinh chinh' if TUNED else 'CHUA tinh chinh'}", font=font, fill=(255, 255, 255))
    d.text((12, 38), f"t = {t_sim:5.2f} s  |  {phase}", font=font, fill=(255, 220, 120))
    d.text((12, 66), info, font=font_s, fill=(200, 230, 255))
    ff.stdin.write(img.tobytes())
    frames += 1
    return img


def run_phase(name, q_list, grip, duration):
    """Noi suy muot (min-jerk) qua cac diem q; ham kep chuyen deu toi goc moi."""
    global t_sim, hold_ref
    q_start = joint_q()
    g_start = joint_q()[5]
    pts = [q_start[:5]] + [q[:5] for q in q_list]
    n_steps = max(1, int(duration * PHYSICS_HZ))
    seg = len(pts) - 1
    for k in range(n_steps):
        s = (k + 1) / n_steps
        s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5  # min-jerk
        u = s * seg
        i = min(int(u), seg - 1)
        a = pts[i] + (u - i) * (pts[i + 1] - pts[i])
        g = g_start + min(1.0, (k + 1) / max(1, int(0.5 * PHYSICS_HZ))) * (grip - g_start)
        set_q(a, g)
        SimulationManager.step(steps=1)
        t_sim += 1.0 / PHYSICS_HZ
        if (k + 1) % steps_per_frame == 0:
            q = joint_q()
            T = kin.tcp(np.r_[q[:5], q[5]], "grasp")
            p, R = cube_pose()
            rel = np.linalg.inv(T) @ np.r_[p, 1.0]
            holding = name in ("nang (MoveL)", "di chuyen", "ha xuong (MoveL)") and grip == GRIP_CLOSE
            if holding and hold_ref is not None:
                slip = np.linalg.norm(rel[:3] - hold_ref) * 1000
                metrics["slip_mm"] = max(metrics["slip_mm"], slip)
                metrics["jitter_mm"].append(rel[:3].copy())
                metrics["tilt_hold_deg"] = max(metrics["tilt_hold_deg"], math.degrees(math.acos(min(1.0, abs(R[2, 2])))))
            info = (f"ham kep {math.degrees(q[5]):5.1f} deg | cube z {p[2] * 1000:5.1f} mm | "
                    f"truot max {metrics['slip_mm']:.1f} mm")
            grab(name, info)
    if name in ("tren cube", "kep"):  # chan doan: FK (URDF) vs pose that trong Isaac, va bam lenh khop
        q = joint_q()
        pw, qw = RigidPrim(jaw_fixed.GetPath().pathString).get_world_poses()
        T_fk = kin.fk(np.r_[q[:5], q[5]], "part_56")
        log(f"DIAG {name}: part_56 Isaac {np.round(pw.numpy()[0] * 1000, 2).tolist()} mm, FK {np.round(T_fk[:3, 3] * 1000, 2).tolist()} mm; "
            f"q dat {np.degrees(q_list[-1][:5]).round(2).tolist()} thuc {np.degrees(q[:5]).round(2).tolist()}")
    if name == "kep":
        q = joint_q()
        T = kin.tcp(np.r_[q[:5], q[5]], "grasp")
        p, _ = cube_pose()
        hold_ref = (np.linalg.inv(T) @ np.r_[p, 1.0])[:3]
        metrics["grip_deg"] = math.degrees(q[5])
        log(f"RESULT grip_closed_deg {metrics['grip_deg']:.2f} cube_in_tcp_mm {(hold_ref * 1000).round(2).tolist()}")


snap_after = {"kep": "02_grasp.png", "di chuyen": "03_carry.png", "nha": "04_released.png"}
for name, q_list, grip, dur in plan:
    run_phase(name, q_list, grip, dur)
    p, _ = cube_pose()
    log(f"PHASE {name:20s} t={t_sim:6.2f}s cube=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}) gripper={math.degrees(joint_q()[5]):.1f}deg")
    if name in snap_after:
        img = grab(name, "")
        if img is not None:
            img.save(os.path.join(OUT, snap_after[name]))
# giu them 1 s de cube nam yen
for _ in range(args.fps):
    SimulationManager.step(steps=steps_per_frame)
    t_sim += steps_per_frame / PHYSICS_HZ
    grab("xong", "")

p, R = cube_pose()
err = np.linalg.norm(p[:2] - PLACE) * 1000
tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
on_ground = abs(p[2] - CUBE / 2) < 0.003
jit = np.std(np.array(metrics["jitter_mm"]), axis=0).max() * 1000 if metrics["jitter_mm"] else float("nan")
ok = err < 10 and tilt < 10 and on_ground and metrics["slip_mm"] < 5
log(f"RESULT place_err_mm {err:.2f} tilt_deg {tilt:.2f} on_ground {on_ground} cube_z_mm {p[2] * 1000:.2f}")
log(f"RESULT slip_max_mm {metrics['slip_mm']:.2f} jitter_std_mm {jit:.3f} tilt_hold_max_deg {metrics['tilt_hold_deg']:.2f}")
log(f"RESULT pick_place {'PASS' if ok else 'FAIL'} tuning={'on' if TUNED else 'off'} frames={frames} video={os.path.join(OUT, 'pick_place_cube.mp4')}")

ff.stdin.close()
ff.wait()
app_utils.stop()
app.close()
