# controllers.py

import torch
import threading
import sys
import os
import time
from abc import ABC, abstractmethod

try:
    from pynput import keyboard
except ImportError:
    print("Warning: pynput is not installed. 'keyboard' control mode will not be available.")
    keyboard = None

try:
    import pygame
except ImportError:
    print("Warning: pygame is not installed. 'joy' control mode will not be available.")
    pygame = None


class BaseController(ABC):
    """
    Abstract base class for controllers with a consistent interface.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self._commands = torch.zeros(1, 3)
        self._stop_event = threading.Event()
        self.thread = None

    @abstractmethod
    def _listener_loop(self):
        """
        Main input loop for the controller, running in a background thread.
        """
        pass

    def start(self):
        """Start the listener loop in a background thread."""
        self.thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.thread.start()
        print(f"\033[92m[INFO] Started {self.__class__.__name__}.\033[0m")

    def stop(self):
        """Stop the listener thread safely."""
        print(f"\n\033[93m[INFO] Stopping {self.__class__.__name__}...\033[0m")
        self._stop_event.set()
        if self.thread:
            self.thread.join()
        print(f"\033[92m[INFO] {self.__class__.__name__} stopped.\033[0m")

    def get_commands(self) -> torch.Tensor:
        """Return the current velocity command."""
        return self._commands.clone()


class RemoteKeyboardController(BaseController):
    """
    Remote control through stdin (terminal). Commands are incremental.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        print("\033[92m       Enter command (w/s/a/d/q/e) and press Enter.\033[0m")
        print("\033[92m       ' ' (space) + Enter to reset velocity.\033[0m")
        print("\033[92m       'exit' + Enter to stop the program.\n\033[0m")

    def _listener_loop(self):
        while not self._stop_event.is_set():
            try:
                user_input = sys.stdin.readline().strip().lower()
                if not user_input: continue

                if 'exit' in user_input:
                    os._exit(0)

                for char in user_input:
                    if char == 'w': self._commands[0, 0] += 0.05
                    elif char == 's': self._commands[0, 0] -= 0.05
                    elif char == 'a': self._commands[0, 1] += 0.05
                    elif char == 'd': self._commands[0, 1] -= 0.05
                    elif char == 'q': self._commands[0, 2] += 0.1
                    elif char == 'e': self._commands[0, 2] -= 0.1
                    elif char == ' ': self._commands[:] = 0.0
                
                self._commands[0, 0] = torch.clamp(self._commands[0, 0], self.cfg.command_minus_x_range, self.cfg.command_plus_x_range)
                self._commands[0, 1] = torch.clamp(self._commands[0, 1], self.cfg.command_minus_y_range, self.cfg.command_plus_y_range)
                self._commands[0, 2] = torch.clamp(self._commands[0, 2], self.cfg.command_minus_yaw_range, self.cfg.command_plus_yaw_range)

            except Exception as e:
                print(f"Error in remote keyboard listener: {e}")
                break


class LocalKeyboardController(BaseController):
    """
    Local keyboard control via pynput. Movement is held while a key is pressed.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        if keyboard is None:
            raise ImportError("pynput is not installed. Cannot use 'keyboard' mode.")
        self.key_states = {'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False}
        self.pynput_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def _on_press(self, key):
        try:
            char = key.char
            if char in self.key_states:
                self.key_states[char] = True
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            char = key.char
            if char in self.key_states:
                self.key_states[char] = False
        except AttributeError:
            if key == keyboard.Key.space:
                self.key_states = {k: False for k in self.key_states}

    def _listener_loop(self):
        self.pynput_listener.start()
        self.pynput_listener.join()

    def stop(self):
        super().stop()
        self.pynput_listener.stop()

    def get_commands(self) -> torch.Tensor:
        current_commands = torch.zeros(1, 3)
        current_commands[0, 0] = 0.1
        if self.key_states['w']: current_commands[0, 0] = self.cfg.command_plus_x_range
        if self.key_states['s']: current_commands[0, 0] = self.cfg.command_minus_x_range
        if self.key_states['a']: current_commands[0, 1] = self.cfg.command_plus_y_range
        if self.key_states['d']: current_commands[0, 1] = self.cfg.command_minus_y_range
        if self.key_states['q']: current_commands[0, 2] = self.cfg.command_plus_yaw_range
        if self.key_states['e']: current_commands[0, 2] = self.cfg.command_minus_yaw_range
        return current_commands


class JoystickController(BaseController):
    """
    Joystick control via pygame. Requires a connected joystick.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        if pygame is None:
            raise ImportError("pygame is not installed. Cannot use 'joy' mode.")
        
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise ConnectionError("No joystick found. Please connect a joystick.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Joystick '{self.joystick.get_name()}' initialized.")
        self.deadzone = 0.1

    def _listener_loop(self):
        while not self._stop_event.is_set():
            pygame.event.get()

            vx_axis = -self.joystick.get_axis(1)
            vy_axis = self.joystick.get_axis(0)
            vyaw_axis = self.joystick.get_axis(3)

            vx = vx_axis if abs(vx_axis) > self.deadzone else 0.0
            vy = vy_axis if abs(vy_axis) > self.deadzone else 0.0
            vyaw = vyaw_axis if abs(vyaw_axis) > self.deadzone else 0.0

            self._commands[0, 0] = vx * (self.cfg.command_plus_x_range if vx > 0 else -self.cfg.command_minus_x_range)
            self._commands[0, 1] = vy * (self.cfg.command_plus_y_range if vy > 0 else -self.cfg.command_minus_y_range)
            self._commands[0, 2] = vyaw * (self.cfg.command_plus_yaw_range if vyaw > 0 else -self.cfg.command_minus_yaw_range)

            time.sleep(0.02)

    def stop(self):
        super().stop()
        pygame.quit()
