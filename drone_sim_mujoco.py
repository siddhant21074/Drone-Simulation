# drone_sim_mujoco.py
# 3D Drone Simulation with orbit camera (Option B), smooth controls, HUD, custom OBJ rendering.
# Expects Models/completo.obj in the project folder.

import os
import sys
import math
import time
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

# -------------------- Configuration --------------------
MODEL_PATH = os.path.join("Models", "drone_costum.obj")

DISPLAY = (1280, 800)
FPS_TARGET = 60

# Physics
GRAVITY = -9.81
DRONE_MASS = 1.6
TIME_STEP = 1.0 / FPS_TARGET
MAX_THRUST = 40.0                 # Newtons total (4 rotors combined)
THROTTLE_STEP = 0.012             # per frame throttle change when key pressed
THROTTLE_DECAY = 0.995           # small decay when not pressing
MAX_ALT_RATE = 6.0               # m/s vertical speed cap
LINEAR_DRAG = 0.8                # simple air drag multiplier per second
ANGULAR_DAMPING = 0.92

# Control gains (tuned for stable, smooth feel)
ROLL_RATE = 60.0                 # degrees/sec per roll input
PITCH_RATE = 60.0
YAW_RATE = 90.0

# Camera (orbit)
CAM_MIN_DIST = 3.0
CAM_MAX_DIST = 60.0
CAM_DEFAULT_DIST = 10.0
CAM_SENS_X = 0.25                 # orbit sensitivity
CAM_SENS_Y = 0.25
CAM_MIN_PITCH = -85.0
CAM_MAX_PITCH = 85.0

# HUD
HUD_BG_ALPHA = 180

# Misc
AUTO_SCALE_FACTOR = 0.9          # final visual tuning for model scale

# -------------------- Robust OBJ loader (center + normalize + display list) --------------------
class SimpleOBJ:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.vertices = []
        self.faces = []
        self.display_list = None
        # load
        self._load_obj(path)
        if len(self.vertices) == 0:
            raise ValueError("OBJ has no vertices")
        self._center_and_normalize()
        self._create_display_list()

    def _load_obj(self, path):
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self.vertices.append((x, y, z))
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    face = []
                    for p in parts:
                        if p == '':
                            continue
                        v = p.split('/')[0]
                        try:
                            idx = int(v) - 1
                        except:
                            idx = None
                        face.append(idx)
                    if len(face) >= 3:
                        self.faces.append(face)
        # debug
        print(f"[OBJ] loaded {len(self.vertices)} vertices, {len(self.faces)} faces from {path}")

    def _center_and_normalize(self):
        v = np.array(self.vertices, dtype=np.float64)
        min_v = v.min(axis=0)
        max_v = v.max(axis=0)
        center = (min_v + max_v) / 2.0
        v = v - center
        max_dim = (max_v - min_v).max()
        if max_dim <= 0:
            scale = 1.0
        else:
            scale = (1.0 / max_dim) * AUTO_SCALE_FACTOR
        v = v * scale
        self.vertices = [tuple(x) for x in v]
        self.bbox = (min_v, max_v)
        # debug
        print(f"[OBJ] centered and scaled (scale={scale:.5f})")

    def _create_display_list(self):
        # Precompile triangles into display list for speed
        try:
            if self.display_list:
                glDeleteLists(self.display_list, 1)
        except Exception:
            pass
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            # fan triangulation
            for i in range(1, len(face) - 1):
                tri = (face[0], face[i], face[i + 1])
                valid = True
                for vi in tri:
                    if vi is None or vi < 0 or vi >= len(self.vertices):
                        valid = False
                        break
                if not valid:
                    continue
                # optional normal approximations could be added, but OpenGL fixed pipeline shading is ok
                v0 = np.array(self.vertices[tri[0]], dtype=np.float32)
                v1 = np.array(self.vertices[tri[1]], dtype=np.float32)
                v2 = np.array(self.vertices[tri[2]], dtype=np.float32)
                # compute face normal
                normal = np.cross(v1 - v0, v2 - v0)
                if np.linalg.norm(normal) > 1e-8:
                    normal = normal / np.linalg.norm(normal)
                    glNormal3f(float(normal[0]), float(normal[1]), float(normal[2]))
                # vertices
                glVertex3f(v0[0], v0[1], v0[2])
                glVertex3f(v1[0], v1[1], v1[2])
                glVertex3f(v2[0], v2[1], v2[2])
        glEnd()
        glEndList()
        self.display_list = dl
        print("[OBJ] display list created")

    def draw(self):
        if self.display_list:
            glCallList(self.display_list)

# -------------------- Drone dynamics (improved controls & stable motion) --------------------
class Drone:
    def __init__(self):
        # position and orientation
        self.pos = np.array([0.0, 2.5, 0.0], dtype=np.float64)   # start a bit higher
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)   # roll, yaw, pitch (degrees)
        self.ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # controls
        self.throttle = 0.0   # 0..1
        self.input_roll = 0.0
        self.input_pitch = 0.0
        self.input_yaw = 0.0
        # misc
        self.sim_time = 0.0

    def update(self, dt):
        # Limit throttle
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        # Total upward thrust (body-up axis)
        thrust_world = np.array([0.0, self.throttle * MAX_THRUST, 0.0], dtype=np.float64)

        # Orientation: compute rotation matrix from roll, pitch, yaw.
        roll = math.radians(self.rot[0])
        yaw = math.radians(self.rot[1])
        pitch = math.radians(self.rot[2])

        # Build rotation: Yaw (Y), Roll (X), Pitch (Z) as earlier
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
        Rz = np.array([[math.cos(pitch), -math.sin(pitch), 0],
                       [math.sin(pitch), math.cos(pitch), 0],
                       [0, 0, 1]])
        Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                       [0, 1, 0],
                       [-math.sin(yaw), 0, math.cos(yaw)]])
        R = Ry @ Rx @ Rz

        # thrust in world coordinates
        thrust_world = R @ np.array([0.0, self.throttle * MAX_THRUST, 0.0])

        # gravity
        gravity = np.array([0.0, DRONE_MASS * GRAVITY, 0.0], dtype=np.float64)
        # drag
        drag = -LINEAR_DRAG * self.vel

        # net force
        net = gravity + thrust_world + drag
        acc = net / DRONE_MASS

        # integrate linear
        self.vel += acc * dt
        # cap vertical speed for predictability
        if self.vel[1] > MAX_ALT_RATE:
            self.vel[1] = MAX_ALT_RATE
        if self.vel[1] < -MAX_ALT_RATE:
            self.vel[1] = -MAX_ALT_RATE

        self.pos += self.vel * dt

        # angular updates (apply control inputs as angular velocities)
        self.ang_vel[0] = self.input_roll * ROLL_RATE    # deg/s
        self.ang_vel[2] = self.input_pitch * PITCH_RATE
        self.ang_vel[1] += self.input_yaw * YAW_RATE * dt   # yaw integrates slowly when input

        # integrate rot (degrees)
        self.rot += self.ang_vel * dt
        # damping for angular motion
        self.ang_vel *= ANGULAR_DAMPING

        # prevent sinking below ground
        if self.pos[1] < 0.1:
            self.pos[1] = 0.1
            if self.vel[1] < 0:
                self.vel[1] = 0.0

        self.sim_time += dt

# -------------------- Camera Orbit (Option B) --------------------
class OrbitCamera:
    def __init__(self, target=None):
        self.target = target
        self.dist = CAM_DEFAULT_DIST
        self.pitch = 20.0
        self.yaw = 45.0
        self.orbiting = True
        self.last_mouse = None
        self.dragging = False

    def handle_mouse(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # left click to drag orbit
                self.dragging = True
                self.last_mouse = pygame.mouse.get_pos()
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
                self.last_mouse = None
        elif event.type == MOUSEWHEEL:
            # zoom with wheel
            self.dist = np.clip(self.dist - event.y * 0.8, CAM_MIN_DIST, CAM_MAX_DIST)

    def update_with_keys(self, keys):
        # arrow keys rotate camera smoothly
        if keys[K_LEFT]:
            self.yaw -= 1.8
        if keys[K_RIGHT]:
            self.yaw += 1.8
        if keys[K_UP]:
            self.pitch = np.clip(self.pitch + 1.2, CAM_MIN_PITCH, CAM_MAX_PITCH)
        if keys[K_DOWN]:
            self.pitch = np.clip(self.pitch - 1.2, CAM_MIN_PITCH, CAM_MAX_PITCH)
        # +/- zoom
        if keys[K_EQUALS] or keys[K_PLUS]:
            self.dist = max(CAM_MIN_DIST, self.dist - 0.5)
        if keys[K_MINUS]:
            self.dist = min(CAM_MAX_DIST, self.dist + 0.5)

    def process_mouse_drag(self):
        if self.dragging and self.last_mouse is not None:
            x, y = pygame.mouse.get_pos()
            lx, ly = self.last_mouse
            dx = x - lx
            dy = y - ly
            self.yaw += dx * CAM_SENS_X
            self.pitch -= dy * CAM_SENS_Y
            self.pitch = np.clip(self.pitch, CAM_MIN_PITCH, CAM_MAX_PITCH)
            self.last_mouse = (x, y)

    def apply_view(self):
        if self.target is None:
            center = np.array([0.0, 0.0, 0.0])
        else:
            center = np.array(self.target.pos, dtype=np.float64)

        # spherical coords
        rad_pitch = math.radians(self.pitch)
        rad_yaw = math.radians(self.yaw)
        x = center[0] + self.dist * math.cos(rad_pitch) * math.sin(rad_yaw)
        y = center[1] + self.dist * math.sin(rad_pitch)
        z = center[2] + self.dist * math.cos(rad_pitch) * math.cos(rad_yaw)

        gluLookAt(x, y, z, center[0], center[1], center[2], 0, 1, 0)

# -------------------- HUD Helper --------------------
class HUD:
    def __init__(self, width, height):
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 18)
        self.width = width
        self.height = height
        self.last_surface = None
        self.last_key = None

    def render(self, drone, wind, camera):
        texts = [
            f"Time: {drone.sim_time:.1f}s",
            f"Altitude: {drone.pos[1]:.2f} m",
            f"Throttle: {drone.throttle:.2f}",
            f"Position: ({drone.pos[0]:.2f}, {drone.pos[1]:.2f}, {drone.pos[2]:.2f})",
            f"Velocity: ({drone.vel[0]:.2f}, {drone.vel[1]:.2f}, {drone.vel[2]:.2f})",
            f"Wind: ({wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}) m/s",
            f"Camera dist: {camera.dist:.1f}, pitch: {camera.pitch:.1f}, yaw: {camera.yaw:.1f}",
        ]
        controls = [
            "CONTROLS:",
            "Enter : Toggle Flight | Space/Shift : Throttle Up/Down",
            "W/S : Pitch forward/back | A/D : Roll left/right | Q/E : Yaw left/right",
            "Mouse drag (left) or Arrow keys : Orbit camera",
            "+/- : Zoom | R : Reset | F : Toggle follow",
            "J/L / I/K / U/O : change wind X/Z/Y"
        ]

        # create surface
        surf_w = 420
        surf_h = max(220, 24 * (len(texts) + len(controls)))
        surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
        surf.fill((10, 10, 10, HUD_BG_ALPHA))

        y = 8
        for t in texts:
            txt = self.font.render(t, True, (230, 230, 230))
            surf.blit(txt, (8, y))
            y += 26

        y += 6
        for t in controls:
            txt = self.small.render(t, True, (200, 220, 255))
            surf.blit(txt, (8, y))
            y += 20

        return surf

# -------------------- Utility functions --------------------
def init_opengl(width, height):
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 8.0, 6.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.95, 0.95, 0.95, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / float(height), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)

# -------------------- Main --------------------
def main():
    pygame.init()
    pygame.display.set_caption("3D Drone Simulation — Orbit Camera (B)")

    screen = pygame.display.set_mode(DISPLAY, DOUBLEBUF | OPENGL)
    init_opengl(DISPLAY[0], DISPLAY[1])
    clock = pygame.time.Clock()

    # load model (if present)
    use_obj = os.path.exists(MODEL_PATH)
    if use_obj:
        try:
            obj_model = SimpleOBJ(MODEL_PATH)
        except Exception as e:
            print("[ERROR] loading OBJ:", e)
            use_obj = False
            obj_model = None
    else:
        obj_model = None
        print("[INFO] OBJ not found, will use dummy if necessary.")

    drone = Drone()
    camera = OrbitCamera(target=drone)
    hud = HUD(DISPLAY[0], DISPLAY[1])

    running = True
    drone_active = False
    gust_enabled = False
    steady_wind = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    last_hud_time = 0.0

    # initial mouse capture state: allow dragging orbit by left button
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)

    # precompute a simple grid geometry? we draw immediate (cheap)
    while running:
        dt = TIME_STEP
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            camera.handle_mouse(event)

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_RETURN:
                    drone_active = not drone_active
                elif event.key == K_r:
                    # reset
                    drone = Drone()
                    camera.target = drone
                    drone_active = False
                    steady_wind = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                elif event.key == K_f:
                    # toggle follow mode (we keep orbit but camera target remains drone)
                    camera.target = drone
                elif event.key == K_g:
                    gust_enabled = not gust_enabled
                # wind adjustments
                elif event.key == K_j:
                    steady_wind[0] -= 0.5
                elif event.key == K_l:
                    steady_wind[0] += 0.5
                elif event.key == K_i:
                    steady_wind[2] += 0.5
                elif event.key == K_k:
                    steady_wind[2] -= 0.5
                elif event.key == K_u:
                    steady_wind[1] += 0.3
                elif event.key == K_o:
                    steady_wind[1] -= 0.3

            elif event.type == MOUSEMOTION:
                # allow camera drag when left button held
                if pygame.mouse.get_pressed()[0]:
                    camera.process_mouse_drag()

            elif event.type == MOUSEWHEEL:
                camera.handle_mouse(event)

        keys = pygame.key.get_pressed()

        # Camera keyboard adjustments
        camera.update_with_keys(keys)

        # Flight control mapping (smooth)
        # throttle up / down
        if keys[K_SPACE]:
            drone.throttle = min(1.0, drone.throttle + THROTTLE_STEP)
        else:
            # gentle decay so drone gently descends if not holding throttle, but allow hover with small thrust
            drone.throttle *= THROTTLE_DECAY

        if keys[K_LSHIFT] or keys[K_RSHIFT]:
            drone.throttle = max(0.0, drone.throttle - THROTTLE_STEP * 1.5)

        # pitch/roll from W A S D (centered when not pressed)
        pitch_in = 0.0
        if keys[K_a]:      # A now moves forward (pitch forward)
            pitch_in += 1.0
        if keys[K_d]:      # D now moves backward (pitch backward)
            pitch_in -= 1.0

        roll_in = 0.0
        if keys[K_w]:      # W now tilts left (roll left)
            roll_in -= 1.0
        if keys[K_s]:      # S now tilts right (roll right)
            roll_in += 1.0

        yaw_in = 0.0
        if keys[K_e]:
            yaw_in += 1.0
        if keys[K_q]:
            yaw_in -= 1.0

        # map inputs into drone smooth inputs (scale)
        drone.input_pitch = float(pitch_in) * 0.9
        drone.input_roll = float(roll_in) * 0.9
        drone.input_yaw = float(yaw_in) * 0.6

        # gusts: occasional random spike when enabled
        wind = steady_wind.copy()
        if gust_enabled and np.random.rand() < 0.01:
            wind += np.random.normal(scale=1.5, size=3)

        # small wind influence on drone velocity (for effect)
        # we incorporate wind by adding small velocity bias; keep stable
        drone.vel += 0.01 * wind * dt

        # Update drone physics
        drone.update(dt)

        # Clamp positions to avoid runaway (optional safety)
        if np.any(np.abs(drone.pos) > 1000):
            drone.pos = np.array([0.0, 2.5, 0.0], dtype=np.float64)
            drone.vel[:] = 0.0

        # Render pass
        glClearColor(0.04, 0.06, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # camera apply (orbit)
        camera.apply_view()

        # draw ground and grid
        # ground quad
        glPushMatrix()
        draw_ground = True
        if draw_ground:
            glDisable(GL_LIGHTING)
            glColor3f(0.07, 0.12, 0.15)
            glBegin(GL_QUADS)
            size = 80
            glVertex3f(-size, 0.0, -size)
            glVertex3f(size, 0.0, -size)
            glVertex3f(size, 0.0, size)
            glVertex3f(-size, 0.0, size)
            glEnd()
            glEnable(GL_LIGHTING)

        # grid
        glDisable(GL_LIGHTING)
        glColor3f(0.25, 0.35, 0.45)
        glBegin(GL_LINES)
        for i in range(-40, 41, 2):
            glVertex3f(i, 0.01, -40)
            glVertex3f(i, 0.01, 40)
            glVertex3f(-40, 0.01, i)
            glVertex3f(40, 0.01, i)
        glEnd()
        glEnable(GL_LIGHTING)
        glPopMatrix()

        # draw drone (obj if available else simple fallback)
        if use_obj and obj_model is not None:
            glPushMatrix()
            # position + orientation (match SimpleOBJ drawn size)
            glTranslatef(drone.pos[0], drone.pos[1], drone.pos[2])
            # apply yaw, roll, pitch order (match physics)
            glRotatef(drone.rot[1], 0, 1, 0)   # yaw
            glRotatef(drone.rot[0], 1, 0, 0)   # roll
            glRotatef(drone.rot[2], 0, 0, 1)   # pitch
            # scale tweak to make sure visible
            glScalef(1.0, 1.0, 1.0)
            obj_model.draw()
            glPopMatrix()
        else:
            # fallback simple cube drone
            glPushMatrix()
            glTranslatef(drone.pos[0], drone.pos[1], drone.pos[2])
            glRotatef(drone.rot[1], 0, 1, 0)
            glRotatef(drone.rot[0], 1, 0, 0)
            glRotatef(drone.rot[2], 0, 0, 1)
            glColor3f(0.2, 0.5, 1.0)
            glScalef(0.4, 0.12, 0.4)
            # cube
            glut_box = False
            if glut_box:
                from OpenGL.GLUT import glutSolidCube
                glutSolidCube(1.0)
            else:
                # simple quad-body
                glBegin(GL_QUADS)
                # top
                glVertex3f(-0.5, 0.5, -0.5)
                glVertex3f(0.5, 0.5, -0.5)
                glVertex3f(0.5, 0.5, 0.5)
                glVertex3f(-0.5, 0.5, 0.5)
                # bottom
                glVertex3f(-0.5, -0.5, -0.5)
                glVertex3f(0.5, -0.5, -0.5)
                glVertex3f(0.5, -0.5, 0.5)
                glVertex3f(-0.5, -0.5, 0.5)
                # sides...
                glEnd()
            glPopMatrix()

        # HUD: render via pygame surface then draw using glDrawPixels
        hud_surface = hud.render(drone, steady_wind, camera)
        hud_data = pygame.image.tostring(hud_surface, "RGBA", True)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, DISPLAY[0], DISPLAY[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glRasterPos2i(8, 8 + hud_surface.get_height())   # draw top-left
        glDrawPixels(hud_surface.get_width(), hud_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, hud_data)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        pygame.display.flip()
        clock.tick(FPS_TARGET)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
