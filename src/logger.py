import subprocess
import psutil
import csv
import time
import os
from datetime import datetime


def get_temperature():
    result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
    return float(result.stdout.strip().replace("temp=", "").replace("'C", ""))


def get_cpu_freq_mhz():
    result = subprocess.run(['vcgencmd', 'measure_clock', 'arm'], capture_output=True, text=True)
    freq_hz = int(result.stdout.strip().split('=')[1])
    return round(freq_hz / 1_000_000, 1)


def get_throttle_status():
    result = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True)
    return result.stdout.strip()


def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / 1024 / 1024, 1)


def get_cpu_percent():
    return psutil.cpu_percent(interval=1)


class ThermalLogger:
    def __init__(self, model_name, runtime_format, cooling_condition, output_dir="results/raw_csv"):
        self.model_name = model_name
        self.runtime_format = runtime_format
        self.cooling_condition = cooling_condition
        self.start_time = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{runtime_format}_{cooling_condition}_{timestamp}.csv"
        self.filepath = os.path.join(output_dir, filename)

        os.makedirs(output_dir, exist_ok=True)

        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'elapsed_seconds',
                'temperature_C',
                'cpu_freq_MHz',
                'throttle_status',
                'fps',
                'ram_usage_MB',
                'cpu_percent'
            ])

        print(f"Logging to: {self.filepath}")

    def start(self):
        self.start_time = time.time()

    def log(self, fps):
        if self.start_time is None:
            self.start()

        elapsed = round(time.time() - self.start_time, 1)

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed,
            get_temperature(),
            get_cpu_freq_mhz(),
            get_throttle_status(),
            round(fps, 2),
            get_ram_usage_mb(),
            get_cpu_percent()
        ]

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[{elapsed}s] Temp: {row[2]}C | Clock: {row[3]}MHz | FPS: {row[5]} | Throttle: {row[4]}")