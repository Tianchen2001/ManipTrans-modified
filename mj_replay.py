"""
Kinematic replay of recorded ManipTrans rollouts in MuJoCo (no physics: every frame each hand link is placed at the
pose recorded in IsaacGym). Input = the *_traj.npz files the evaluators already write (eval/sweep_ep1000/).

Run from the ManipTrans root on a GPU-less or GPU node (headless: MUJOCO_GL=egl, or MUJOCO_GL=osmesa without a GPU):

  # one DD / Ours pair -> side-by-side snapshot image + two videos
  MUJOCO_GL=egl python mj_replay.py --pair h_dd_seq0_raw_inspire h_ours_seq0_raw_inspire
  MUJOCO_GL=egl python mj_replay.py --pair h_dd_bseq1_raw_bih_inspire h_ours_bseq1_raw_bih_inspire --snapshots 5

Videos play at --speed 0.5 (half speed) by default.
Everything at once: --all (= --strip --trail + side-by-side video). Filmstrip: --strip (--strip-only: nothing else)
Paths: --trail
Two hands: contact frames are printed; --contact-frames centres the snapshots on the DD contact; orange dot = contact
Options: --env 0 (which recorded rollout, 0..3)  --azim 135 --elev -25 (camera)  --snapshots 4 (hands in the image)
         --frames 0 40 80 115 (explicit snapshot frames)  --no-video  --size 1280 960  --color-dd / --color-ours (hex)
Effort: --effort wrist|joint|total adds a power-vs-time panel under the side-by-side video (cursor = current frame),
         under the filmstrip with --strip (<dd run>__vs__ours_strip_effort.png/.pdf, time axis aligned with the columns),
         and writes <dd run>__vs__ours_effort.png/.pdf (solid = rendered rollout --env, band = 25-75 % over all rollouts)
Methods: dd (Data-driven), hx (HandX), ours; colours --color-dd / --color-hx / --color-ours
Outputs (vis/mj/): <run>.mp4, <run>_snap.png, <dd run>__vs__ours.png (side by side, same camera)
"""
import argparse, os, re
import xml.etree.ElementTree as ET
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs=2, metavar=("DD_RUN", "OURS_RUN"), help="two run names (without _traj.npz)")
ap.add_argument("--runs", nargs="+", default=[], help="single runs (rendered separately)")
ap.add_argument("--traj-dir", default="eval/sweep_ep1000")
ap.add_argument("--assets", default="maniptrans_envs/assets")
ap.add_argument("--out", default="vis/mj")
ap.add_argument("--env", type=int, default=0)
ap.add_argument("--azim", type=float, default=135.0)
ap.add_argument("--elev", type=float, default=-25.0)
ap.add_argument("--zoom", type=float, default=1.0, help=">1 = closer")
ap.add_argument("--snapshots", type=int, default=6)
ap.add_argument("--frames", type=int, nargs="+", default=None)
ap.add_argument("--contact-frames", action="store_true",
                help="two hands: centre the snapshot frames on the contact of the first (DD) run instead of spreading them")
ap.add_argument("--size", type=int, nargs=2, default=[640, 480])
ap.add_argument("--fps", type=int, default=30, help="video fps")
ap.add_argument("--speed", type=float, default=0.5, help="playback speed of the videos (0.5 = half speed)")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--separate-videos", action="store_true", help="with --pair: also write one video per run")
ap.add_argument("--artimano-style", choices=["robot", "skin"], default="robot",
                help="robot: per-link collision meshes + joint spheres (reads as an articulated robot hand); skin: MANO surface")
ap.add_argument("--joint-radius", type=float, default=0.006)
ap.add_argument("--strip", action="store_true",
                help="filmstrip: one panel per snapshot frame (camera follows the hand), DD row above Ours row")
ap.add_argument("--strip-only", action="store_true", help="only the filmstrip (no overlay image / video)")
ap.add_argument("--all", action="store_true", help="everything: filmstrip + overlay image with paths + video(s)")
ap.add_argument("--strip-size", type=int, nargs=2, default=[600, 600], help="render size of one panel before cropping")
ap.add_argument("--strip-width", type=int, default=1000, help="final figure width in px (single column ~ 1000 px at 300 dpi)")
ap.add_argument("--strip-font", type=int, default=None,
                help="text height in px in the final figure (default 26 single hand, 20 two hands; 30 px ~ 7 pt at 300 dpi)")
ap.add_argument("--metrics-csv", default="eval/summary/sweep_summary_ep1000.csv",
                help="sweep summary; with a pair, the table metrics where Ours is better go top-left in the video")
ap.add_argument("--no-metrics", action="store_true")
ap.add_argument("--video-font", type=int, default=25, help="text size (px) in videos / overlay images, as main_video.py")
ap.add_argument("--label-scale", type=float, default=1.0, help="size of the method / time labels in the overlay image and video")
ap.add_argument("--strip-floor", action="store_true", help="keep the floor/shadows in the filmstrip (default: plain white)")
ap.add_argument("--strip-fixed", action="store_true", help="filmstrip with one fixed camera instead of following the hand")
ap.add_argument("--trail", action="store_true",
                help="overlay image: draw the fingertip and wrist paths over the whole clip (jitter shows as zigzag)")
ap.add_argument("--color-dd", default="#6FD1D7")
ap.add_argument("--color-ours", default="#FF4081")
ap.add_argument("--color-hx", default="#e09f3e")
ap.add_argument("--light-ambient", type=float, default=0.28, help="headlight ambient (was 0.5: lighter, flatter)")
ap.add_argument("--light-diffuse", type=float, default=0.55, help="directional (top) light diffuse (was 0.45)")
ap.add_argument("--hand-specular", type=float, default=0.12, help="hand material specular (was 0.25)")
ap.add_argument("--hand-shade", type=float, default=0.9,
                help="hand mesh colour = method colour x this (labels and curves keep the exact colour); 1 = old look")
ap.add_argument("--curve-lw", type=float, default=1.8, help="line width (pt) of the effort curves")
ap.add_argument("--export", default=None, metavar="DIR",
                help="export the raw material of the filmstrip instead of composing it: per pair one npz with the "
                     "snapshot panels rendered in neutral white (+ hand masks) and the per-frame effort statistics, "
                     "so that colours, layout and curves can be changed afterwards without re-rendering")
ap.add_argument("--effort", choices=["none", "wrist", "joint", "total"], default="none",
                help="add an effort-vs-time panel under the side-by-side video (+ a static figure), from "
                     "<traj-dir>/<run>_frames.npz: wrist = wrist actuation power (P_wrist of the tables), "
                     "joint = finger joints, total = both; summed over both hands")
ap.add_argument("--effort-smooth", type=int, default=5, help="moving-average window in frames (1 = raw)")
ap.add_argument("--effort-height", type=float, default=0.3, help="effort panel height as a fraction of the video height")
ap.add_argument("--strip-effort-height", type=float, default=0.24,
                help="filmstrip: height of the effort curve under the strip, as a fraction of --strip-width")
a = ap.parse_args()
if a.all:
    a.strip, a.trail = True, True
STEP = max(1, round(60 * a.speed / a.fps))   # sim steps (60 Hz) per video frame
FPS = 60 * a.speed / STEP                   # exact playback speed

import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

URDF = {  # (hand, side) -> urdf relative to assets
    ("inspire", "rh"): "inspire_hand/inspire_hand_right.urdf", ("inspire", "lh"): "inspire_hand/inspire_hand_left.urdf",
    ("artimano", "rh"): "mano_urdf/rh_mano.urdf", ("artimano", "lh"): "mano_urdf/lh_mano.urdf",
    ("shadow", "rh"): "shadow_hand/shadow_hand_right_woarm.urdf", ("shadow", "lh"): "shadow_hand/shadow_hand_left_woarm.urdf",
}
NAME = re.compile(r"h_(?P<method>[a-z]+)_(?P<clip>[a-z]*\d+)_(?P<lab>[a-z0-9]+)(?:_(?P<side>bih|rh|lh))?_(?P<hand>[a-z]+)")


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def rpy_to_quat(r, p, y):                                   # URDF fixed-axis rpy -> MuJoCo wxyz
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def urdf_visuals(path, tag="visual"):
    """link name -> list of (mesh file, scale[3], pos[3], quat_wxyz[4]) from the <visual> (or <collision>) meshes"""
    root, base = ET.parse(path).getroot(), os.path.dirname(path)
    out = {}
    for link in root.findall("link"):
        for v in link.findall(tag):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            o = v.find("origin")
            xyz = [float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            sc = [float(x) for x in m.get("scale", "1 1 1").split()]
            f = os.path.normpath(os.path.join(base, m.get("filename").replace("package://", "")))
            out.setdefault(link.get("name"), []).append((f, sc, xyz, rpy_to_quat(*rpy)))
    return out


def load_run(run):
    m = NAME.match(run)
    assert m, f"cannot parse run name {run}"
    D = np.load(f"{a.traj_dir}/{run}_traj.npz")
    rb = D["rb"][:, a.env]                                  # [T, nb, 13]: pos, quat xyzw, vel
    alive = D["alive"][:, a.env].astype(bool)
    T = int(alive.sum()) if alive.any() else len(alive)
    hands = []                                              # (side, names, rb_idx)
    if "body_names" in D.files:
        hands.append(("rh", list(D["body_names"]), D["body_rb_idx"]))
    else:
        for s in ("rh", "lh"):
            hands.append((s, list(D[f"body_names_{s}"]), D[f"body_rb_idx_{s}"]))
    contact = D["contact"][:T, a.env] if "contact" in D.files else None   # two hands: net hand-hand contact force (N)
    if contact is not None:
        on = np.nonzero(contact > 0.1)[0]
        print(f"{run}: hand-hand contact (>0.1 N) in {len(on)}/{T} frames" +
              (f", peak {contact.max():.2f} N at frame {int(contact.argmax())}, first {on[0]}, last {on[-1]}" if len(on) else ""))
    return dict(run=run, method=m["method"], hand=m["hand"], clip=m["clip"], rb=rb[:T], T=T, hands=hands, contact=contact)


def build_xml(runs_copies, floor_z, w, h, ctr=(0, 0, 0), ext=1.0, extra="", floor=True):
    """runs_copies: list of (prefix, run dict, rgba). One mocap body per hand link and copy."""
    assets, bodies, meshes = [], [], {}
    for prefix, R, rgba in runs_copies:
        for side, names, _ in R["hands"]:
            robot = R["hand"] == "artimano" and a.artimano_style == "robot"
            vis = urdf_visuals(os.path.join(a.assets, URDF[(R["hand"], side)]), "collision" if robot else "visual")
            for nm in names:
                geoms = []
                for k, (f, sc, xyz, q) in enumerate(vis.get(nm, [])):
                    key = (f, tuple(sc))
                    if key not in meshes:
                        meshes[key] = f"m{len(meshes)}"
                        assets.append(f'<mesh name="{meshes[key]}" file="{f}" scale="{sc[0]} {sc[1]} {sc[2]}"/>')
                    geoms.append(f'<geom type="mesh" mesh="{meshes[key]}" pos="{xyz[0]} {xyz[1]} {xyz[2]}" '
                                 f'quat="{q[0]} {q[1]} {q[2]} {q[3]}" rgba="{rgba[0] * a.hand_shade:.3f} {rgba[1] * a.hand_shade:.3f} {rgba[2] * a.hand_shade:.3f} 1" '
                                 f'contype="0" conaffinity="0" material="handmat"/>')
                if robot and vis.get(nm):        # joint marker: dark sphere at the link origin (= the joint)
                    geoms.append(f'<geom type="sphere" size="{a.joint_radius}" rgba="0.15 0.15 0.15 1" '
                                 f'contype="0" conaffinity="0" material="handmat"/>')
                bodies.append(f'<body name="{prefix}_{side}_{nm}" mocap="true">{"".join(geoms)}</body>')
    return f"""<mujoco>
  <statistic center="{ctr[0]} {ctr[1]} {ctr[2]}" extent="{ext}"/>
  <visual><global offwidth="{w}" offheight="{h}"/><quality shadowsize="8192" offsamples="8"/>
    <map znear="0.01" shadowclip="1.5" shadowscale="0.6"/>
    <headlight ambient="{a.light_ambient} {a.light_ambient} {a.light_ambient}" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/></visual>
  <asset>{''.join(assets)}
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.95 0.95 0.95" rgb2="0.92 0.92 0.92" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="16 16" reflectance="0.0"/>
    <material name="handmat" specular="{a.hand_specular}" shininess="0.3"/></asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0.15 0.1 -1" diffuse="{a.light_diffuse} {a.light_diffuse} {a.light_diffuse}" castshadow="true"/>
    {'<geom type="plane" size="3 3 0.1" pos="0 0 %s" material="floor" contype="0" conaffinity="0"/>' % floor_z if floor else ''}
    {''.join(bodies)}
    {extra}
  </worldbody></mujoco>"""


def set_pose(model, data, prefix, R, t):
    for side, names, idx in R["hands"]:
        for nm, i in zip(names, idx):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{side}_{nm}")
            if bid < 0 or model.body_mocapid[bid] < 0:
                continue
            mid = model.body_mocapid[bid]
            p, q = R["rb"][t, i, :3], R["rb"][t, i, 3:7]
            data.mocap_pos[mid] = p
            data.mocap_quat[mid] = [q[3], q[0], q[1], q[2]]   # xyzw -> wxyz


def all_positions(R):
    return np.concatenate([R["rb"][:, idx, :3].reshape(-1, 3) for _, _, idx in R["hands"]])


def camera(P):
    lo, hi = P.min(0), P.max(0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = (np.linalg.norm(hi - lo) + 0.2) * 1.0 / a.zoom
    cam.azimuth, cam.elevation = a.azim, a.elev
    return cam


def render(xml, poses, cam, w, h):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for prefix, R, t in poses:
        set_pose(model, data, prefix, R, t)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, h, w)
    r.update_scene(data, camera=cam)
    img = r.render()
    r.close()
    return img


# video text in the style of main_video.py: bold DejaVu (fallbacks), 1-px black shadow, PAD from the edges
PAD, SHADOW = 12, (0, 0, 0)


def video_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def put_text(img, text, corner, color):
    """corner: 'br' (bottom right) or 'bl' (bottom left)"""
    im = Image.fromarray(np.ascontiguousarray(img[..., :3]))
    d = ImageDraw.Draw(im)
    F = video_font(int(round(a.video_font * a.label_scale)))
    tw = d.textlength(text, font=F)
    asc, desc = F.getmetrics()                        # same baseline for every label, with or without descenders
    x = img.shape[1] - tw - PAD if corner == "br" else PAD
    y = img.shape[0] - (asc + desc) - PAD
    d.text((x + 1, y + 1), text, font=F, fill=SHADOW)
    d.text((x, y), text, font=F, fill=color)
    return np.asarray(im)


DIVIDER_W, DIVIDER_C = 4, 180                      # wall between the two videos, as main_video.py

# table metrics: csv column, text, unit, format, higher is better
# (column, text, unit, format, higher is better, always shown)
METRICS = [("sr_auc", "AUC\u2191", "", "{:.2f}", True, True),
           ("x_power_wrist_W", "P_wrist\u2193", " W", "{:.2f}", False, True)]


def metric_lines(G):
    """per run: lines of the metrics where Ours (last run) is strictly better than the baseline (after rounding)"""
    if a.no_metrics or len(G) != 2 or not os.path.exists(a.metrics_csv):
        if not a.no_metrics and len(G) == 2:
            print("no metrics overlay: missing", a.metrics_csv)
        return [[], []]
    import csv
    rows = {r["exp"]: r for r in csv.DictReader(open(a.metrics_csv))}
    if any(R["run"] not in rows for R in G):
        print("no metrics overlay: run not in", a.metrics_csv)
        return [[], []]
    out = [[], []]
    for col, name, unit, f, hi, always in METRICS:
        try:
            v = [float(f.format(float(rows[R["run"]][col]))) for R in G]
        except (KeyError, ValueError):
            continue
        if always or ((v[1] > v[0]) if hi else (v[1] < v[0])):
            for i in range(2):
                out[i].append(f"{name}: {f.format(v[i])}{unit}")
    print("metrics overlay:", out[1] if out[1] else "none (Ours not better on any)")
    return out


def overlay_top_left(img, lines, color):
    """as _overlay_top_left in main_video.py"""
    if not lines:
        return img
    im = Image.fromarray(np.ascontiguousarray(img[..., :3]))
    d = ImageDraw.Draw(im)
    F = video_font(int(round(a.video_font * a.label_scale)))
    col = tuple(int(255 * c) for c in color)
    y = PAD
    asc, desc = F.getmetrics()
    for ln in lines:                                  # fixed line pitch (main_video uses the bbox -> uneven)
        d.text((PAD + 1, y + 1), ln, font=F, fill=SHADOW)
        d.text((PAD, y), ln, font=F, fill=col)
        y += asc + desc + 2
    return np.asarray(im)


def divider(h):
    return np.full((h, DIVIDER_W, 3), DIVIDER_C, np.uint8)


def label(img, text, color):
    return put_text(img, text, "br", tuple(int(255 * c) for c in color))


def trail_xml(R, rgb):
    """small spheres along the wrist (first body) and fingertip paths over the whole clip"""
    g = []
    for side, names, idx in R["hands"]:
        sel = [0] + [k for k, nm in enumerate(names) if "tip" in nm.lower()]
        for k in sel:
            P = R["rb"][:, idx[k], :3]
            for t in range(0, len(P)):
                c = rgb * 0.75
                g.append(f'<geom type="sphere" size="0.0022" pos="{P[t,0]:.4f} {P[t,1]:.4f} {P[t,2]:.4f}" '
                         f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 0.9" contype="0" conaffinity="0"/>')
    return "".join(g)


def frame_label(img, text):
    return put_text(img, text, "bl", (90, 90, 90))


def render_strip(R, floor_z, gcam, neutral=False):
    """one rendered panel per snapshot frame (uncropped, no labels); neutral: white hands + hand masks"""
    w, h = a.strip_size
    rgba = np.full(3, NEUTRAL) / a.hand_shade if neutral else method_color(R)   # neutral grey, headroom for recolouring
    xml = build_xml([("v", R, rgba)], floor_z, w, h, ctr=tuple(gcam.lookat), ext=max(0.3, gcam.distance),
                    floor=a.strip_floor and not neutral)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rr = mujoco.Renderer(model, h, w)
    sr = mujoco.Renderer(model, h, w) if neutral else None
    if sr is not None:
        sr.enable_segmentation_rendering()
    panels, masks = [], []
    for t in snapshot_frames(R["T"]):
        set_pose(model, data, "v", R, t)
        mujoco.mj_forward(model, data)
        if a.strip_fixed:
            cam = gcam
        else:                                                  # follow: centre on the hand(s) at frame t
            Pt = np.concatenate([R["rb"][t, idx, :3] for _, _, idx in R["hands"]])
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = (Pt.min(0) + Pt.max(0)) / 2
            cam.distance = (np.linalg.norm(Pt.max(0) - Pt.min(0)) + 0.12) * 1.25 / a.zoom
            cam.azimuth, cam.elevation = a.azim, a.elev
        rr.update_scene(data, camera=cam)
        panels.append(rr.render().copy())
        if sr is not None:
            sr.update_scene(data, camera=cam)
            masks.append((sr.render()[..., 0] >= 0).astype(np.uint8))
    rr.close()
    if sr is not None:
        sr.close()
        return panels, masks
    return panels


NEUTRAL = 0.7


def export_pair(G, floor_z, cam):
    """raw material for an offline figure: neutral panels, masks, snapshot frames, effort statistics"""
    os.makedirs(a.export, exist_ok=True)
    fr = snapshot_frames(min(R["T"] for R in G))
    out = {"frames": np.array(fr), "fps": 60, "env": a.env, "methods": np.array([R["method"] for R in G]),
           "runs": np.array([R["run"] for R in G]), "azim": a.azim, "elev": a.elev, "neutral": NEUTRAL}
    for R in G:
        m = R["method"]
        p, k = render_strip(R, floor_z, cam, neutral=True)
        out[f"panels_{m}"], out[f"masks_{m}"] = np.stack(p), np.stack(k)
        if R["contact"] is not None:
            out[f"contact_{m}"] = R["contact"]
        fpath = f"{a.traj_dir}/{R['run']}_frames.npz"
        if os.path.exists(fpath):
            D = np.load(fpath)
            alive = D["alive"].astype(bool) if "alive" in D.files else None
            for comp in ("power_wrist_W", "power_joint_W"):
                keys = [f for f in D.files if f.startswith("m_") and f.endswith(comp)]
                if not keys:
                    continue
                P = sum(D[f].astype(np.float64) for f in keys)                    # [T, rollouts], summed over hands
                if alive is not None:
                    P = np.where(alive, P, np.nan)
                c = comp.replace("power_", "").replace("_W", "")
                out[f"{c}_env_{m}"] = P[:, a.env]
                for q in (10, 25, 50, 75, 90):
                    out[f"{c}_q{q}_{m}"] = np.nanpercentile(P, q, axis=1)
                out[f"{c}_mean_{m}"] = np.nanmean(P, axis=1)
    path = f"{a.export}/{G[0]['run']}__vs__ours.npz"
    np.savez_compressed(path, **out)
    print("exported", path, {k: v.shape for k, v in out.items() if hasattr(v, "shape") and v.ndim > 1})


def compose_strip(rows, names, colors, times, contacts=None):
    """crop all panels to one common box, add a time header and row labels, scale to --strip-width"""
    allp = np.stack([p for r in rows for p in r])
    ink = (allp.astype(int).min(-1) < 245).any(0)              # union of non-white pixels
    ys, xs = np.nonzero(ink)
    m = 6
    y0, y1, x0, x1 = max(ys.min() - m, 0), ys.max() + m, max(xs.min() - m, 0), xs.max() + m
    rows = [[p[y0:y1, x0:x1] for p in r] for r in rows]
    ph, pw = rows[0][0].shape[:2]
    ncol, gap = len(rows[0]), 4
    # text size fixed in the FINAL figure (--strip-font px at --strip-width), independent of the number of panels
    Wwo = ncol * pw + (ncol - 1) * gap
    two = any(len(R_["hands"]) == 2 for R_ in STRIP_RUNS)
    F, SW = a.strip_font or (20 if two else 26), a.strip_width
    def font(bold, size):
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf", size)
        except OSError:
            return ImageFont.load_default()
    # row labels: vertical (rotated) if the longest name fits the row height, otherwise horizontal on the left
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    L = max(nm.count("\n") + 1 for nm in names)                                   # lines per label
    tl1 = max(probe.textlength(ln, font=font(True, 100)) for nm in names for ln in nm.split("\n")) / 100
    fs = int(F * Wwo / (SW - 1.9 * F))
    fl = min(1.1 * fs, 0.92 * ph / tl1)              # vertical: shrink the label font until the longest line fits
    vertical = fl >= 0.75 * 1.1 * fs
    if not vertical:
        kh = 1.1 * tl1 + 0.6
        fs = int(F * Wwo / (SW - kh * F))
        fl, lab_w = 1.1 * fs, int(kh * fs)
    else:
        lab_w = int(fl * (0.6 + 1.15 * L))
    head_h = int(1.6 * fs)
    W = lab_w + Wwo
    H = head_h + len(rows) * ph + (len(rows) - 1) * gap
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    Ft, Fl = font(False, fs), font(True, int(fl))
    for c, t in enumerate(times):
        txt = f"{t / 60:.2f}s"
        x = lab_w + c * (pw + gap) + pw / 2 - d.textlength(txt, font=Ft) / 2
        d.text((x, head_h - fs - 6), txt, fill=(60, 60, 60), font=Ft)
    for r, (row, nm, col) in enumerate(zip(rows, names, colors)):
        y = head_h + r * (ph + gap)
        for c, p in enumerate(row):
            im.paste(Image.fromarray(p), (lab_w + c * (pw + gap), y))
            if contacts is not None and contacts[r] is not None and contacts[r][c] > 0.1:   # hand-hand contact marker
                rad = int(fs * 0.3)
                cx, cy = lab_w + c * (pw + gap) + pw - rad - 2, y + rad + 2
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(230, 160, 0))
        rgb = tuple(int(255 * v) for v in col)
        lines = nm.split("\n")
        if vertical:
            lab = Image.new("RGB", (ph, lab_w), (255, 255, 255))      # vertical row label (first line leftmost)
            ld = ImageDraw.Draw(lab)
            y0l = (lab_w - len(lines) * 1.15 * fl) / 2
            for i, ln in enumerate(lines):
                ld.text(((ph - ld.textlength(ln, font=Fl)) / 2, y0l + i * 1.15 * fl), ln, fill=rgb, font=Fl)
            im.paste(lab.rotate(90, expand=True), (0, y))
        else:                                                         # horizontal, right-aligned next to the row
            y0l = y + ph / 2 - len(lines) * 1.15 * fl / 2
            for i, ln in enumerate(lines):
                d.text((lab_w - d.textlength(ln, font=Fl) - 0.3 * fs, y0l + i * 1.15 * fl), ln, fill=rgb, font=Fl)
    sc = a.strip_width / W
    STRIP_GEOM.update(centers=[(lab_w + c * (pw + gap) + pw / 2) * sc for c in range(ncol)], lab_w=lab_w * sc, F=F)
    return im.resize((a.strip_width, int(H * sc)), Image.LANCZOS)


def effort_strip(G, fr):
    """effort curve under the filmstrip; with evenly spaced snapshots the time axis is aligned so that each snapshot
    time sits under the centre of its column"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ps = [load_effort(R) for R in G]
    T = min([min(R["T"] for R in G)] + [len(P) for P in Ps])
    W, dpi = a.strip_width, 300
    F = STRIP_GEOM["F"]                                   # text height in px, same as the strip labels
    pt = F * 72 / dpi
    bot, top = 3.4 * F, 2.0 * F                           # room for tick labels + x label / legend (px)
    H = max(int(a.strip_effort_height * W), int(bot + top + 4 * F))
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fr = [min(t, T - 1) for t in fr]
    d = np.diff(fr)
    if len(fr) > 1 and np.all(np.abs(d - d[0]) <= 1):    # aligned with the columns
        x0, x1 = STRIP_GEOM["centers"][0] / W, STRIP_GEOM["centers"][-1] / W
        tmin, tmax = fr[0] / 60, fr[-1] / 60
    else:
        x0, x1 = (STRIP_GEOM["lab_w"] + 0.06 * W) / W, 0.98
        tmin, tmax = 0, (T - 1) / 60
    ax = fig.add_axes([x0, bot / H, x1 - x0, 1 - (bot + top) / H])
    t = np.arange(T) / 60
    for R, P in zip(G, Ps):
        col = method_color(R)
        ax.fill_between(t, np.nanpercentile(P[:T], 25, 1), np.nanpercentile(P[:T], 75, 1), color=col, alpha=0.18, lw=0)
        ax.plot(t, P[:T, a.env], color=col, lw=a.curve_lw, label=f"{one_line(SHORT.get(R['method'], R['method']))}: "
                                                          f"mean {np.nanmean(P[:T]):.2f} W")
    for k in fr:                                          # snapshot times
        ax.axvline(k / 60, color="0.7", lw=0.5, ls="--", zorder=0)
    ax.set_xlim(tmin, tmax); ax.set_ylim(bottom=0)
    ax.set_xticks([k / 60 for k in fr]); ax.set_xticklabels(["" if k == 0 else f"{k / 60:.2f}" for k in fr])  # no "0"
    ax.set_xlabel("time (s)", fontsize=pt, labelpad=1)
    ax.set_ylabel({"wrist": "wrist power (W)", "joint": "finger power (W)", "total": "power (W)"}[a.effort],
                  fontsize=pt, labelpad=3)
    ax.tick_params(labelsize=pt * 0.9, width=0.8, length=3, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(G), fontsize=pt * 0.9,
              handlelength=1.2, borderaxespad=0.0, columnspacing=1.0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(img[:, :W])


CONTACT_REF = None
STRIP_RUNS = []
STRIP_GEOM = {}


def side_by_side_video(G, cam, floor_z):
    """DD (left) and Ours (right) in one video, same camera, method labels and time stamp"""
    w, h = a.size
    sc = dict(ctr=tuple(cam.lookat), ext=max(0.3, cam.distance))
    rend = []
    for R in G:
        model = mujoco.MjModel.from_xml_string(build_xml([("v", R, method_color(R))], floor_z, w, h, **sc))
        rend.append((R, model, mujoco.MjData(model), mujoco.Renderer(model, h, w)))
    T = min(R["T"] for R in G)
    panel = None
    if a.effort != "none":
        Wv = 2 * w + DIVIDER_W
        panel, xs, (y0, y1), T = effort_panel(G, T, Wv, int(a.effort_height * h))
    ML = metric_lines(G)
    frames = []
    for t in range(0, T, STEP):
        tiles = []
        for R, model, data, rr in rend:
            set_pose(model, data, "v", R, t)
            mujoco.mj_forward(model, data)
            rr.update_scene(data, camera=cam)
            tile = label(rr.render(), NAMES.get(R["method"], R["method"]), method_color(R))
            tiles.append(overlay_top_left(tile, ML[len(tiles)], method_color(R)))
        frame = np.concatenate([tiles[0], divider(h), tiles[1]], 1)
        frame = frame_label(frame, f"t = {t / 60:.2f} s")
        if panel is not None:                               # effort panel with a cursor at the current frame
            im = Image.fromarray(panel.copy())
            ImageDraw.Draw(im).line([(xs[t], y0), (xs[t], y1)], fill=(40, 40, 40), width=max(2, w // 400))
            frame = np.concatenate([frame[..., :3], np.asarray(im)], 0)
        frames.append(frame)
    for *_, rr in rend:
        rr.close()
    out = f"{a.out}/{G[0]['run']}__vs__ours{'_effort' if panel is not None else ''}.mp4"
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=1)
    print("wrote", out)


def snapshot_frames(T):
    if a.frames:
        return a.frames
    if a.contact_frames and CONTACT_REF is not None and (CONTACT_REF > 0.1).any():
        on = np.nonzero(CONTACT_REF[:T] > 0.1)[0]
        lo, hi = max(0, on[0] - 15), min(T - 1, on[-1] + 15)     # contact window +- 0.25 s
        return np.linspace(lo, hi, a.snapshots).round().astype(int).tolist()
    return np.linspace(0, T - 1, a.snapshots).round().astype(int).tolist()


NAMES = {"dd": "Geometry only", "hx": "HandX", "ours": "Ours"}
SHORT = {"dd": "Geo", "hx": "HandX", "ours": "Ours"}     # strip row labels + legend; "\n" -> 2 lines
def one_line(s):
    return s.replace("\n", " ")


def method_color(R):
    return hexrgb({"dd": a.color_dd, "hx": a.color_hx}.get(R["method"], a.color_ours))


EFF_KEYS = {"wrist": ["power_wrist_W"], "joint": ["power_joint_W"], "total": ["power_wrist_W", "power_joint_W"]}


def load_effort(R):
    """[T, rollouts] actuation power summed over the hands (from the sweep per-frame dump)"""
    D = np.load(f"{a.traj_dir}/{R['run']}_frames.npz")
    P = 0
    for k in EFF_KEYS[a.effort]:            # two hands: m_rh_<k> + m_lh_<k>; single hand: m_<k> (or m_rh_<k>)
        keys = [f for f in D.files if f.startswith("m_") and f.endswith(k)]
        if not keys:
            raise KeyError(f"{R['run']}_frames.npz has no per-frame '{k}' (keys: {sorted(D.files)})")
        for f in keys:
            P = P + D[f].astype(np.float64)
    if "alive" in D.files:
        P = np.where(D["alive"].astype(bool), P, np.nan)
    if a.effort_smooth > 1:
        ker = np.ones(a.effort_smooth) / a.effort_smooth
        P = np.stack([np.convolve(np.nan_to_num(P[:, i]), ker, mode="same") for i in range(P.shape[1])], 1)
    return P


def effort_axes(G, Ps, T, fig_w, fig_h, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(T) / 60
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    for R, P in zip(G, Ps):
        col = method_color(R)
        ax.fill_between(t, np.nanpercentile(P[:T], 25, 1), np.nanpercentile(P[:T], 75, 1), color=col, alpha=0.18, lw=0)
        ax.plot(t, P[:T, a.env], color=col, lw=1.8, label=f"{NAMES.get(R['method'], R['method'])} "
                                                           f"(mean {np.nanmean(P[:T]):.2f} W)")
    ax.set_xlim(0, t[-1]); ax.set_ylim(bottom=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel({"wrist": "wrist power (W)", "joint": "finger power (W)", "total": "actuation power (W)"}[a.effort])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig, ax


def effort_panel(G, T, width, height):
    """static figure + (panel image, x pixel of each frame, y range of the plot area) for the video"""
    import matplotlib.pyplot as plt
    Ps = [load_effort(R) for R in G]
    T = min([T] + [len(P) for P in Ps])
    fig, _ = effort_axes(G, Ps, T, 6.0, 2.4, 200)
    stem = f"{a.out}/{G[0]['run']}__vs__ours_effort"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf | mean " + ", ".join(f"{NAMES.get(R['method'])}: {np.nanmean(P[:T]):.2f} W"
                                                      for R, P in zip(G, Ps)))
    dpi = 100
    fig, ax = effort_axes(G, Ps, T, width / dpi, height / dpi, dpi)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    H = img.shape[0]
    ylo, yhi = ax.get_ylim()
    y0, y1 = [H - v for v in ax.transData.transform([(0, yhi), (0, ylo)])[:, 1]]
    xs = ax.transData.transform(np.stack([np.arange(T) / 60, np.zeros(T)], 1))[:, 0]
    plt.close(fig)
    if img.shape[1] < width:                                 # match the video width exactly
        img = np.concatenate([img, np.full((img.shape[0], width - img.shape[1], 3), 255, np.uint8)], 1)
    return img[:, :width], xs, (y0, y1), T


def render_run(R, cam, floor_z, video=True):
    sc = dict(ctr=tuple(cam.lookat), ext=max(0.3, cam.distance))
    w, h = a.size
    base = method_color(R)
    fr = snapshot_frames(R["T"])
    # snapshot: earlier frames lighter (blended towards white), last frame full colour
    copies = [(f"s{k}", R, base * (0.35 + 0.65 * (k + 1) / len(fr)) + (1 - (0.35 + 0.65 * (k + 1) / len(fr))))
              for k in range(len(fr))]
    snap = render(build_xml(copies, floor_z, w, h, **sc, extra=trail_xml(R, base) if a.trail else ""),
                  [(f"s{k}", R, t) for k, t in enumerate(fr)], cam, w, h)
    if video and not a.no_video:
        xml = build_xml([("v", R, base)], floor_z, w, h, **sc)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        rr = mujoco.Renderer(model, h, w)
        vid = []
        for t in range(0, R["T"], STEP):
            set_pose(model, data, "v", R, t)
            mujoco.mj_forward(model, data)
            rr.update_scene(data, camera=cam)
            vid.append(rr.render())
        rr.close()
        imageio.mimsave(f"{a.out}/{R['run']}.mp4", vid, fps=FPS, macro_block_size=1)
    Image.fromarray(snap).save(f"{a.out}/{R['run']}_snap.png")
    print(f"wrote {a.out}/{R['run']}_snap.png" + ("" if (a.no_video or not video) else f" and .mp4 (frames {fr})"))
    return snap


os.makedirs(a.out, exist_ok=True)
groups = ([[load_run(r) for r in a.pair]] if a.pair else []) + [[load_run(r)] for r in a.runs]
for G in groups:
    CONTACT_REF = G[0]["contact"]
    P = np.concatenate([all_positions(R) for R in G])
    cam = camera(P)                                         # same camera for both methods of a pair
    floor_z = float(P[:, 2].min()) - 0.03
    if a.export:
        export_pair(G, floor_z, cam)
        continue
    if a.strip:
        STRIP_RUNS = G
        rows = [render_strip(R, floor_z, cam) for R in G]
        fr = snapshot_frames(min(R["T"] for R in G))
        fig = compose_strip(rows, [SHORT.get(R["method"], R["method"]) for R in G], [method_color(R) for R in G],
                            fr, [None if R["contact"] is None else R["contact"][fr] for R in G])
        out = f"{a.out}/{G[0]['run']}{'__vs__ours' if len(G) == 2 else ''}_strip.png"
        if a.effort != "none":                                # effort curve under the filmstrip
            cur = effort_strip(G, fr)
            both = Image.new("RGB", (fig.size[0], fig.size[1] + cur.size[1]), (255, 255, 255))
            both.paste(fig, (0, 0)); both.paste(cur, (0, fig.size[1]))
            fig = both
            out = out.replace("_strip.png", "_strip_effort.png")
        fig.save(out, dpi=(300, 300))
        fig.save(out.replace(".png", ".pdf"), resolution=300)
        print("wrote", out, fig.size)
        if a.strip_only:
            continue
    imgs = [label(render_run(R, cam, floor_z, video=len(G) == 1 or a.separate_videos),
                  NAMES.get(R["method"], R["method"]), method_color(R)) for R in G]
    if len(G) == 2 and not a.no_video:
        side_by_side_video(G, cam, floor_z)
    if len(G) == 2:
        out = f"{a.out}/{G[0]['run']}__vs__ours.png"
        ML = metric_lines(G)
        imgs = [overlay_top_left(im, ML[i], method_color(R)) for i, (im, R) in enumerate(zip(imgs, G))]
        Image.fromarray(np.concatenate([imgs[0][..., :3], divider(imgs[0].shape[0]), imgs[1][..., :3]], 1)).save(out)
        print("wrote", out)
