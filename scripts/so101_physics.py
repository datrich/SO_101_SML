#!/usr/bin/env python3
"""
Bổ sung thông số vật lý SO-101 (theo TheRobotStudio/SO-ARM100, Simulation/SO101/so101_new_calib.*)
vào URDF do onshape-to-robot xuất ra. Chạy tự động qua "post_import_commands" trong config.json.

- Khối lượng mỗi link: lấy từ so101_new_calib.urdf gốc.
- Trọng tâm + tensor quán tính: tính từ mesh collision của chính model này (giả định mật độ đồng nhất,
  scale theo khối lượng gốc) -> khớp với hệ trục link của model Onshape của bạn.
- Khớp: <dynamics damping/friction> theo class "sts3215" trong so101_new_calib.xml (MuJoCo gốc),
  thêm <transmission> giống URDF gốc. Effort/velocity đặt trong config.json (joint_properties).
- KHÔNG đổi joint limit (giữ nguyên theo Onshape).

Dùng: python so101_physics.py path/to/robot.urdf
"""
import sys
import os
import logging
import numpy as np
import xml.etree.ElementTree as ET
from stl import mesh as stl_mesh

# Mesh ghép từ nhiều part kín -> numpy-stl báo nhầm "not closed"; tích phân thể tích vẫn đúng
logging.getLogger("stl").setLevel(logging.ERROR)

# link trong URDF của bạn -> (tên link gốc, khối lượng gốc kg)
# Đổi key nếu bạn đổi tên link (mate connector "link_xxx") trong Onshape.
LINK_MASS = {
    "base_arm": ("base_link", 0.147),
    "part_31": ("shoulder_link", 0.100006),
    "part_37": ("upper_arm_link", 0.103),
    "part_42": ("lower_arm_link", 0.104),
    "part_63": ("wrist_link", 0.079),
    "part_56": ("gripper_link", 0.087),
    "moving_jaw_so101_v1": ("moving_jaw_so101_v1_link", 0.012),
}

# Servo STS3215 (class "sts3215" trong so101_new_calib.xml)
JOINT_DAMPING = 0.60  # N·m·s/rad
JOINT_FRICTION = 0.052  # N·m (frictionloss)
# armature=0.028, kp=998.22, kv=2.731, forcerange=±2.94 -> không có trong URDF chuẩn, áp khi import sim

MOTOR_ID = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def rpy_to_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_origin(elem):
    o = elem.find("origin")
    xyz = np.zeros(3) if o is None else np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
    rpy = np.zeros(3) if o is None else np.array([float(v) for v in o.get("rpy", "0 0 0").split()])
    return xyz, rpy_to_matrix(*rpy)


def link_mass_properties(link, urdf_dir):
    """Ghép tất cả mesh collision (trong hệ trục link) rồi tính thể tích, trọng tâm, quán tính (mật độ 1)."""
    vectors = []
    for col in link.findall("collision"):
        m = col.find("geometry/mesh")
        if m is None:
            continue
        fn = m.get("filename").replace("package://", "")
        path = os.path.join(urdf_dir, fn)
        xyz, R = parse_origin(col)
        v = stl_mesh.Mesh.from_file(path).vectors.reshape(-1, 3) @ R.T + xyz
        vectors.append(v.reshape(-1, 3, 3))
    if not vectors:
        return None
    data = np.zeros(sum(len(v) for v in vectors), dtype=stl_mesh.Mesh.dtype)
    data["vectors"] = np.concatenate(vectors)
    combined = stl_mesh.Mesh(data, remove_empty_areas=False)
    return combined.get_mass_properties()  # volume, cog, inertia@cog (density=1)


def main(urdf_path):
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    robot = tree.getroot()

    for link in robot.findall("link"):
        name = link.get("name")
        if name not in LINK_MASS:
            continue
        orig_name, mass = LINK_MASS[name]
        volume, cog, inertia = link_mass_properties(link, urdf_dir)
        I = inertia * (mass / volume)

        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
        for child in list(inertial):
            inertial.remove(child)
        ET.SubElement(inertial, "origin", xyz="%.6g %.6g %.6g" % tuple(cog), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="%g" % mass)
        ET.SubElement(
            inertial, "inertia",
            ixx="%.6g" % I[0, 0], ixy="%.6g" % I[0, 1], ixz="%.6g" % I[0, 2],
            iyy="%.6g" % I[1, 1], iyz="%.6g" % I[1, 2], izz="%.6g" % I[2, 2],
        )
        print(f"  [inertial] {name:22s} <- {orig_name:26s} m={mass:.4f} kg  com={np.round(cog, 4)}")

    for joint in robot.findall("joint"):
        if joint.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        for old in joint.findall("dynamics") + joint.findall("joint_properties"):
            joint.remove(old)
        ET.SubElement(joint, "dynamics", damping="%g" % JOINT_DAMPING, friction="%g" % JOINT_FRICTION)
        print(f"  [dynamics] {joint.get('name'):22s} damping={JOINT_DAMPING} friction={JOINT_FRICTION}")

    for t in robot.findall("transmission"):
        robot.remove(t)
    for jname, mid in MOTOR_ID.items():
        if robot.find(f"joint[@name='{jname}']") is None:
            continue
        t = ET.SubElement(robot, "transmission", name=f"{jname}_trans")
        ET.SubElement(t, "type").text = "transmission_interface/SimpleTransmission"
        j = ET.SubElement(t, "joint", name=jname)
        ET.SubElement(j, "hardwareInterface").text = "hardware_interface/PositionJointInterface"
        a = ET.SubElement(t, "actuator", name=f"motor{mid}")
        ET.SubElement(a, "hardwareInterface").text = "hardware_interface/PositionJointInterface"
        ET.SubElement(a, "mechanicalReduction").text = "1"

    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"* so101_physics: updated {urdf_path}")


if __name__ == "__main__":
    main(sys.argv[1])
