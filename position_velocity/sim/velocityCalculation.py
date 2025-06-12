from collections import deque

class VelocityTracker:
    
    def init(self, fps=60, frame_interval=5, smoothing_window=5):
        self.fps = fps
        self.frame_interval = frame_interval
        self.dt = frame_interval / fps
        self.prev_pos = None
        self.velocity_buffer = deque(maxlen=smoothing_window)

    def update_position(self, new_pos):
        """
        new_pos: tuple (x, y)
        Returns: smoothed velocity (vx, vy)
        """
        if self.prev_pos is None:
            self.prev_pos = new_pos
            return (0.0, 0.0)  # No velocity yet

        # Compute raw velocity
        dx = new_pos[0] - self.prev_pos[0]
        dy = new_pos[1] - self.prev_pos[1]
        vx = dx / self.dt
        vy = dy / self.dt

        # Update previous position
        self.prev_pos = new_pos

        # Add to buffer and compute moving average
        self.velocity_buffer.append((vx, vy))
        smoothed_vx = sum(v[0] for v in self.velocity_buffer) / len(self.velocity_buffer)
        smoothed_vy = sum(v[1] for v in self.velocity_buffer) / len(self.velocity_buffer)

        return (smoothed_vx, smoothed_vy) 
