import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys

# ---------------------
# Drone & env constants
# ---------------------
GRAVITY = -9.81
DRONE_MASS = 1.5
ARM_LENGTH = 0.18
MAX_THRUST_PER_ROTOR = 10.0
TIME_STEP = 1/60.0

BATTERY_CAPACITY_AH = 2.2
BATTERY_VOLTAGE = 11.1
BATTERY_WH = BATTERY_CAPACITY_AH * BATTERY_VOLTAGE
POWER_COEFF = 0.5
THROTTLE_STEP = 0.03
CONTROL_ROLL_GAIN = 3.5
CONTROL_PITCH_GAIN = 3.5
CONTROL_YAW_GAIN = 1.2
UNLIMITED_BATTERY = True
DEFAULT_WIND = np.array([0.0, 0.0, 0.0])
MAX_WIND_SPEED = 15.0
WIND_STEP = 0.4

# Camera controls
camera_distance = 8.0
camera_angle_x = 30.0
camera_angle_y = 45.0
camera_follow = True

# ---------------------
# Climate model
# ---------------------
def compute_wind_force(velocity, wind_vector, drag_coeff=1.05, area=0.1, air_density=1.225):
    rel = wind_vector - velocity
    speed = np.linalg.norm(rel)
    if speed <= 1e-6:
        return np.zeros(3)
    drag_mag = 0.5 * air_density * speed**2 * drag_coeff * area
    drag_dir = -rel / (speed + 1e-12)
    return drag_mag * drag_dir

# ---------------------
# Drone class (3D)
# ---------------------
class Drone:
    def __init__(self):
        self.pos = np.array([0.0, 2.0, 0.0])  # x, y(height), z
        self.vel = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw (degrees)
        self.ang_vel = np.array([0.0, 0.0, 0.0])
        
        # Control inputs
        self.throttle = 0.0
        self.roll_input = 0.0
        self.pitch_input = 0.0
        self.yaw_input = 0.0
        
        # Battery
        self.battery_wh = BATTERY_WH
        
        # PID for altitude
        self.alt_Kp = 8.0
        self.alt_Kd = 3.0
        self.prev_alt_err = 0.0
        
        self.sim_time = 0.0
        
        # Rotor positions (body frame)
        self.rotors = [
            np.array([ARM_LENGTH, 0, ARM_LENGTH]),    # front-right
            np.array([-ARM_LENGTH, 0, ARM_LENGTH]),   # front-left
            np.array([-ARM_LENGTH, 0, -ARM_LENGTH]),  # back-left
            np.array([ARM_LENGTH, 0, -ARM_LENGTH]),   # back-right
        ]
        
    def get_rotation_matrix(self):
        """Convert roll, pitch, yaw to rotation matrix"""
        roll, pitch, yaw = np.radians(self.rotation)
        
        # Roll (X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (Z-axis) 
        Rz = np.array([
            [np.cos(pitch), -np.sin(pitch), 0],
            [np.sin(pitch), np.cos(pitch), 0],
            [0, 0, 1]
        ])
        
        # Yaw (Y-axis)
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        return Ry @ Rx @ Rz
        
    def update(self, dt, wind_vector, gust_enabled=True):
        # Desired altitude
        desired_alt = 2.0 + self.throttle * 3.0
        alt_err = desired_alt - self.pos[1]
        alt_der = (alt_err - self.prev_alt_err) / dt
        self.prev_alt_err = alt_err
        alt_output = self.alt_Kp * alt_err + self.alt_Kd * alt_der
        
        # Total thrust
        total_thrust = np.clip(DRONE_MASS * (9.81 + alt_output), 0, MAX_THRUST_PER_ROTOR * 4)
        
        # Distribute thrust with control inputs
        thrusts = np.ones(4) * (total_thrust / 4.0)
        
        # Roll control (tilt left/right)
        thrusts[0] += self.roll_input * CONTROL_ROLL_GAIN  # front-right
        thrusts[3] += self.roll_input * CONTROL_ROLL_GAIN  # back-right
        thrusts[1] -= self.roll_input * CONTROL_ROLL_GAIN  # front-left
        thrusts[2] -= self.roll_input * CONTROL_ROLL_GAIN  # back-left
        
        # Pitch control (tilt forward/back)
        thrusts[0] += self.pitch_input * CONTROL_PITCH_GAIN  # front-right
        thrusts[1] += self.pitch_input * CONTROL_PITCH_GAIN  # front-left
        thrusts[2] -= self.pitch_input * CONTROL_PITCH_GAIN  # back-left
        thrusts[3] -= self.pitch_input * CONTROL_PITCH_GAIN  # back-right
        
        # Yaw control (rotation)
        thrusts[0] += self.yaw_input * CONTROL_YAW_GAIN
        thrusts[2] -= self.yaw_input * CONTROL_YAW_GAIN
        
        thrusts = np.maximum(thrusts, 0)
        
        # Battery model
        if UNLIMITED_BATTERY:
            self.battery_wh = BATTERY_WH
            battery_factor = 1.0
        else:
            power_per_rotor = POWER_COEFF * thrusts * np.sqrt(np.maximum(thrusts, 1e-6))
            total_power = np.sum(power_per_rotor)
            energy_used_wh = (total_power * dt) / 3600.0
            self.battery_wh = max(0.0, self.battery_wh - energy_used_wh)
            battery_factor = self.battery_wh / BATTERY_WH if BATTERY_WH > 0 else 0.0
        thrusts = thrusts * battery_factor
        
        # Wind with gusts
        wind = wind_vector.copy()
        if gust_enabled and np.random.rand() < 0.02:
            gust = np.random.normal(scale=3.0, size=3)
            wind += gust
        
        wind_force = compute_wind_force(self.vel, wind, drag_coeff=1.2, area=0.15)
        
        # Get rotation matrix
        R = self.get_rotation_matrix()
        
        # Thrust in body frame (upward)
        thrust_body = np.array([0, np.sum(thrusts), 0])
        thrust_world = R @ thrust_body
        
        # Net force
        gravity_force = np.array([0.0, DRONE_MASS * GRAVITY, 0.0])
        net_force = thrust_world + gravity_force + wind_force
        
        # Update dynamics
        accel = net_force / DRONE_MASS
        self.vel += accel * dt
        self.pos += self.vel * dt
        
        # Update rotation based on inputs
        self.rotation[0] += self.roll_input * 30 * dt  # roll
        self.rotation[2] += self.pitch_input * 30 * dt  # pitch
        self.rotation[1] += self.yaw_input * 50 * dt  # yaw
        
        # Damping
        self.rotation *= 0.95
        self.rotation = np.clip(self.rotation, -45, 45)
        
        # Ground collision
        if self.pos[1] < 0.1:
            self.pos[1] = 0.1
            self.vel[1] = max(0, self.vel[1])
        
        self.sim_time += dt

# ---------------------
# 3D Drawing functions
# ---------------------
def draw_grid(size=20, step=2):
    glColor3f(0.25, 0.35, 0.4)
    glBegin(GL_LINES)
    for i in range(-size, size + 1, step):
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
    glEnd()

def draw_ground(size=60):
    glPushMatrix()
    glColor3f(0.07, 0.18, 0.15)
    glBegin(GL_QUADS)
    glVertex3f(-size, 0, -size)
    glVertex3f(size, 0, -size)
    glVertex3f(size, 0, size)
    glVertex3f(-size, 0, size)
    glEnd()
    glPopMatrix()

def draw_axes():
    glLineWidth(3)
    glBegin(GL_LINES)
    # X axis - Red
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(2, 0, 0)
    # Y axis - Green
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 2, 0)
    # Z axis - Blue
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 2)
    glEnd()
    glLineWidth(1)

def draw_drone(drone):
    glPushMatrix()
    glTranslatef(drone.pos[0], drone.pos[1], drone.pos[2])
    glRotatef(drone.rotation[1], 0, 1, 0)  # yaw
    glRotatef(drone.rotation[0], 1, 0, 0)  # roll
    glRotatef(drone.rotation[2], 0, 0, 1)  # pitch
    
    # Drone body color based on battery
    battery_pct = drone.battery_wh / BATTERY_WH if not UNLIMITED_BATTERY else 1.0
    glColor3f(0.15 + 0.4 * battery_pct, 0.15, 0.2 + 0.6 * battery_pct)
    
    # Body (box)
    glBegin(GL_QUADS)
    # Top
    glVertex3f(-0.14, 0.03, -0.14)
    glVertex3f(0.14, 0.03, -0.14)
    glVertex3f(0.14, 0.03, 0.14)
    glVertex3f(-0.14, 0.03, 0.14)
    # Bottom
    glVertex3f(-0.12, -0.03, -0.12)
    glVertex3f(0.12, -0.03, -0.12)
    glVertex3f(0.12, -0.03, 0.12)
    glVertex3f(-0.12, -0.03, 0.12)
    # Sides
    glVertex3f(-0.12, -0.03, -0.12)
    glVertex3f(-0.12, 0.03, -0.12)
    glVertex3f(-0.12, 0.03, 0.12)
    glVertex3f(-0.12, -0.03, 0.12)
    
    glVertex3f(0.12, -0.03, -0.12)
    glVertex3f(0.12, 0.03, -0.12)
    glVertex3f(0.12, 0.03, 0.12)
    glVertex3f(0.12, -0.03, 0.12)
    glEnd()
    
    # Draw arms and rotors
    for rotor in drone.rotors:
        # Arm
        glColor3f(0.6, 0.6, 0.65)
        glBegin(GL_QUADS)
        glVertex3f(0, -0.01, 0)
        glVertex3f(rotor[0], -0.01, rotor[2])
        glVertex3f(rotor[0], 0.01, rotor[2])
        glVertex3f(0, 0.01, 0)
        glEnd()
        
        # Rotor (circle)
        glPushMatrix()
        glTranslatef(rotor[0], rotor[1], rotor[2])
        glColor3f(0.9, 0.3, 0.2)
        
        # Draw rotor disc
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for angle in range(0, 361, 30):
            rad = np.radians(angle)
            glVertex3f(0.06 * np.cos(rad), 0, 0.06 * np.sin(rad))
        glEnd()
        glPopMatrix()
    
    glPopMatrix()

def draw_wind_indicator(wind, pos=(15, 0, 0)):
    glPushMatrix()
    glTranslatef(pos[0], pos[1] + 2, pos[2])
    
    # Arrow shaft
    glColor3f(0.5, 1.0, 0.5)
    glLineWidth(3)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(wind[0] * 0.5, wind[1] * 0.5, wind[2] * 0.5)
    glEnd()
    
    # Arrow head
    glTranslatef(wind[0] * 0.5, wind[1] * 0.5, wind[2] * 0.5)
    glColor3f(0.3, 0.8, 0.3)
    glBegin(GL_TRIANGLES)
    glVertex3f(0, 0, 0)
    glVertex3f(-0.2, 0, -0.1)
    glVertex3f(-0.2, 0, 0.1)
    glEnd()
    
    glPopMatrix()
    glLineWidth(1)

def blit_text_surface(text_surface, x, y):
    """Draw a pygame surface as 2D overlay via glDrawPixels."""
    if text_surface is None:
        return
    surface = text_surface.convert_alpha()
    text_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glRasterPos2f(x, y)
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

def draw_overlay_panel(x, y, width, height, color=(0, 0, 0, 0.5)):
    """Draw semi-transparent rectangle for HUD backgrounds."""
    glColor4f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()

# ---------------------
# Initialize Pygame + OpenGL
# ---------------------
def main():
    global camera_distance, camera_angle_x, camera_angle_y, camera_follow
    
    pygame.init()
    display = (1400, 900)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Drone Climate Simulation")
    
    # OpenGL setup
    glClearColor(0.04, 0.06, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Font for HUD
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    
    # Initialize
    drone = Drone()
    steady_wind = DEFAULT_WIND.copy()
    gust_enabled = False
    drone_active = False
    
    clock = pygame.time.Clock()
    running = True
    mouse_down = False
    last_mouse_pos = None
    
    print("=" * 60)
    print("3D DRONE CLIMATE SIMULATION")
    print("=" * 60)
    print("FLIGHT CONTROLS:")
    print("  ENTER: Toggle motors on/off")
    print("  W/S: Pitch forward/backward")
    print("  A/D: Roll left/right")
    print("  Q/E: Yaw left/right")
    print("  SPACE / SHIFT: Throttle up/down")
    print()
    print("CAMERA CONTROLS:")
    print("  Arrow Keys: Rotate camera")
    print("  +/-: Zoom in/out")
    print("  F: Toggle follow mode")
    print()
    print("OTHER:")
    print("  R: Reset simulation")
    print("  G: Toggle gusts")
    print("  Wind adjust: J/L (X axis), I/K (Z axis), U/O (Y axis)")
    print("  ESC: Quit")
    print("=" * 60)
    
    while running:
        dt = TIME_STEP
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    drone = Drone()
                    drone_active = False
                    steady_wind = DEFAULT_WIND.copy()
                elif event.key == pygame.K_f:
                    camera_follow = not camera_follow
                elif event.key == pygame.K_g:
                    gust_enabled = not gust_enabled
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    camera_distance = max(3, camera_distance - 1)
                elif event.key == pygame.K_MINUS:
                    camera_distance = min(30, camera_distance + 1)
                elif event.key == pygame.K_RETURN:
                    drone_active = not drone_active
                    if not drone_active:
                        drone.throttle = 0.0
                        drone.roll_input = 0.0
                        drone.pitch_input = 0.0
                        drone.yaw_input = 0.0
                        drone.vel[:] = 0.0
                elif event.key == pygame.K_BACKSPACE:
                    drone_active = False
                    drone.throttle = 0.0
                    drone.roll_input = 0.0
                    drone.pitch_input = 0.0
                    drone.yaw_input = 0.0
                    drone.vel[:] = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_down = True
                    last_mouse_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down and last_mouse_pos:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    camera_angle_y += dx * 0.5
                    camera_angle_x += dy * 0.5
                    camera_angle_x = np.clip(camera_angle_x, -89, 89)
                    last_mouse_pos = event.pos
        
        # Keyboard controls
        keys = pygame.key.get_pressed()
        
        # Throttle
        if keys[pygame.K_SPACE]:
            drone.throttle = min(1.0, drone.throttle + THROTTLE_STEP)
        elif keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            drone.throttle = max(0.0, drone.throttle - THROTTLE_STEP)

        # Direct axis controls for better responsiveness
        pitch_axis = 0.0
        if keys[pygame.K_w]:
            pitch_axis += 1.0
        if keys[pygame.K_s]:
            pitch_axis -= 1.0
        drone.pitch_input = pitch_axis if drone_active else 0.0

        roll_axis = 0.0
        if keys[pygame.K_d]:
            roll_axis += 1.0
        if keys[pygame.K_a]:
            roll_axis -= 1.0
        drone.roll_input = roll_axis if drone_active else 0.0

        yaw_axis = 0.0
        if keys[pygame.K_e]:
            yaw_axis += 1.0
        if keys[pygame.K_q]:
            yaw_axis -= 1.0
        drone.yaw_input = yaw_axis if drone_active else 0.0

        # Wind controls (I/K = forward/back (Z), J/L = left/right (X), U/O = up/down (Y))
        wind_delta = np.zeros(3)
        if keys[pygame.K_j]:
            wind_delta[0] -= WIND_STEP
        if keys[pygame.K_l]:
            wind_delta[0] += WIND_STEP
        if keys[pygame.K_i]:
            wind_delta[2] += WIND_STEP
        if keys[pygame.K_k]:
            wind_delta[2] -= WIND_STEP
        if keys[pygame.K_u]:
            wind_delta[1] += WIND_STEP
        if keys[pygame.K_o]:
            wind_delta[1] -= WIND_STEP
        if np.any(wind_delta):
            steady_wind = np.clip(steady_wind + wind_delta,
                                  -MAX_WIND_SPEED,
                                  MAX_WIND_SPEED)
        
        # Camera controls (arrow keys)
        if keys[pygame.K_LEFT]:
            camera_angle_y -= 2
        if keys[pygame.K_RIGHT]:
            camera_angle_y += 2
        if keys[pygame.K_UP]:
            camera_angle_x = min(89, camera_angle_x + 2)
        if keys[pygame.K_DOWN]:
            camera_angle_x = max(-89, camera_angle_x - 2)
        
        # Update drone only when active
        if drone_active:
            drone.update(dt, steady_wind, gust_enabled)
        else:
            drone.vel *= 0.98
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera positioning
        if camera_follow:
            # Follow drone
            camera_x = drone.pos[0] + camera_distance * np.sin(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))
            camera_y = drone.pos[1] + camera_distance * np.sin(np.radians(camera_angle_x))
            camera_z = drone.pos[2] + camera_distance * np.cos(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))
            gluLookAt(camera_x, camera_y, camera_z,
                     drone.pos[0], drone.pos[1], drone.pos[2],
                     0, 1, 0)
        else:
            # Fixed camera
            camera_x = camera_distance * np.sin(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))
            camera_y = camera_distance * np.sin(np.radians(camera_angle_x))
            camera_z = camera_distance * np.cos(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))
            gluLookAt(camera_x, camera_y, camera_z, 0, 2, 0, 0, 1, 0)
        
        # Draw scene
        draw_ground()
        draw_grid()
        draw_drone(drone)
        draw_wind_indicator(steady_wind)
        
        # Render HUD using Pygame
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Create text surfaces
        if UNLIMITED_BATTERY:
            battery_pct = 100.0
            battery_line = "Battery: ∞ (Unlimited)"
        else:
            battery_pct = (drone.battery_wh / BATTERY_WH) * 100
            battery_line = f"Battery: {battery_pct:.1f}% ({drone.battery_wh:.2f}Wh)"
        texts = [
            f"Time: {drone.sim_time:.1f}s",
            f"Altitude: {drone.pos[1]:.2f}m",
            f"Position: ({drone.pos[0]:.1f}, {drone.pos[1]:.1f}, {drone.pos[2]:.1f})",
            f"Throttle: {drone.throttle:.2f}",
            battery_line,
            f"Wind: ({steady_wind[0]:.1f}, {steady_wind[1]:.1f}, {steady_wind[2]:.1f}) m/s",
            f"Gusts: {'ON' if gust_enabled else 'OFF'}",
            f"Camera: {'FOLLOW' if camera_follow else 'FIXED'}",
            f"Flight: {'ACTIVE' if drone_active else 'PAUSED'}",
        ]
        panel_height = len(texts) * 25 + 20
        draw_overlay_panel(5, 5, 360, panel_height)

        # Draw text (convert to OpenGL texture)
        y = 10
        for text in texts:
            color = (255, 255, 255) if battery_pct > 20 else (255, 100, 100)
            text_surface = small_font.render(text, True, color)
            blit_text_surface(text_surface, 10, y)
            y += 25

        # Battery warning (only if finite battery)
        if not UNLIMITED_BATTERY and battery_pct < 20:
            warning = font.render("⚠ LOW BATTERY ⚠", True, (255, 50, 50))
            blit_text_surface(warning, display[0]//2 - 100, 30)

        # Controls legend on the right-hand side
        control_lines = [
            "BASIC CONTROLS",
            "Enter : start / pause",
            "Space / Shift : up / down",
            "W S : forward / back tilt",
            "A D : left / right tilt",
            "Q E : yaw rotate",
            "Arrow keys : camera orbit",
            "+ / - : zoom , F : follow cam",
            "Wind axes : J L / I K / U O",
            "R : reset scene",
            "ESC : quit",
        ]
        panel_height_controls = len(control_lines) * 22 + 20
        x = display[0] - 340
        draw_overlay_panel(x - 5, 5, 335, panel_height_controls)
        y = 10
        for line in control_lines:
            text_surface = small_font.render(line, True, (200, 220, 255))
            blit_text_surface(text_surface, x, y)
            y += 22

        if not drone_active:
            stop_msg = font.render("DRONE PAUSED - PRESS ENTER TO FLY", True, (255, 255, 255))
            sub_msg = small_font.render("Set throttle with Space / Shift, ESC to exit", True, (200, 200, 200))
            panel_w = max(stop_msg.get_width(), sub_msg.get_width()) + 40
            panel_h = stop_msg.get_height() + sub_msg.get_height() + 30
            panel_x = (display[0] - panel_w) / 2
            panel_y = display[1] - panel_h - 40
            draw_overlay_panel(panel_x, panel_y, panel_w, panel_h, (0.1, 0.1, 0.1, 0.75))
            blit_text_surface(stop_msg, panel_x + 20, panel_y + 10)
            blit_text_surface(sub_msg, panel_x + 20, panel_y + 20 + stop_msg.get_height())
        
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()