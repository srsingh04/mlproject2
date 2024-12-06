If the timer resets at a specific frame (e.g., 10801), the subsequent frame is 4 seconds after the previous one (10800th) because one frame is missing. However, since we are analyzing on a timescale of minutes, this small gap can be safely ignored.

Need to change frame timer from resetting at 10800 to just 10801. 